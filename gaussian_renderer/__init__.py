#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel,GaussianModel_exr,GaussianModel_color
from utils.sh_utils import eval_sh
from utils.general_utils import flip_align_view
from scene.NVDIFFREC import extract_env_map

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        # debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    mask = pc.get_mask
    means3D = pc.get_xyz * mask
    means2D = screenspace_points
    opacity = pc.get_opacity * mask

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # print(pc.get_features.shape)
            shs = pc.get_features
            
    else:
        colors_precomp = override_color
    # print("SHS",shs.shape)
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    ng = 256
    diff = pc._features_dc
    diff = diff.reshape(ng,ng,3)
    dt = diff[:,:,:3].reshape(-1,1,3)
    
    df = dt.transpose(1,2).flatten(start_dim=1).contiguous()
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    raster_settings_alpha = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor([0,0,0], dtype=torch.float32, device="cuda"),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            # debug=False
        )
    rasterizer_alpha = GaussianRasterizer(raster_settings=raster_settings_alpha)
    alpha = torch.ones_like(means3D) 
    alpha_img =  rasterizer_alpha(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = alpha,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)[0]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "alpha_img":alpha_img,
            "scales":scales,
            "diffuse":df}


def render_spec(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, mlp_color,
           scaling_modifier=1.0, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation


    shs = None
    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0) 
    colors_precomp = mlp_color #+ torch.clamp_min(sh2rgb + 0.5, 0.0) 
    

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)

def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
    cam_z = ndc_xyz[..., 2:3]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())
    return cam_xyz





def depth2point_cam(sampled_depth, ref_intrinsic):
    B, N, C, H, W = sampled_depth.shape
    valid_z = sampled_depth
    valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (W - 1)
    valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (H - 1)
    valid_y, valid_x = torch.meshgrid(valid_y, valid_x)
    # B,N,H,W
    valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
    valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
    ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)  # 1, 1, 5, 512, 640, 3
    cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H) # 1, 1, 5, 512, 640, 3
    return ndc_xyz, cam_xyz


def depth2point_world(depth_image, intrinsic_matrix, extrinsic_matrix):
    # depth_image: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    _, xyz_cam = depth2point_cam(depth_image[None,None,None,...], intrinsic_matrix[None,...])
    xyz_cam = xyz_cam.reshape(-1,3)
    xyz_world = torch.cat([xyz_cam, torch.ones_like(xyz_cam[...,0:1])], axis=-1) @ torch.inverse(extrinsic_matrix).transpose(0,1)
    xyz_world = xyz_world[...,:3]

    return xyz_world

def depth_pcd2normal(xyz):
    hd, wd, _ = xyz.shape 
    bottom_point = xyz[..., 2:hd,   1:wd-1, :]
    top_point    = xyz[..., 0:hd-2, 1:wd-1, :]
    right_point  = xyz[..., 1:hd-1, 2:wd,   :]
    left_point   = xyz[..., 1:hd-1, 0:wd-2, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point 
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(xyz_normal.permute(2,0,1), (1,1,1,1), mode='constant').permute(1,2,0)
    return xyz_normal


def normal_from_depth_image(depth, intrinsic_matrix, extrinsic_matrix):
    # depth: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    # xyz_normal: (H, W, 3)
    xyz_world = depth2point_world(depth, intrinsic_matrix, extrinsic_matrix) # (HxW, 3)
    xyz_world = xyz_world.reshape(*depth.shape, 3)
    xyz_normal = depth_pcd2normal(xyz_world)

    return xyz_normal

def render_normal_ref(viewpoint_cam, depth, bg_color=None, alpha=None):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()

    normal_ref = normal_from_depth_image(depth, intrinsic_matrix.to(depth.device), extrinsic_matrix.to(depth.device))
    background = bg_color[None,None,...]
    normal_ref = normal_ref*alpha[...,None] + background*(1. - alpha[...,None])
    normal_ref = normal_ref.permute(2,0,1)

    return normal_ref

def normalize_normal_inplace(normal, alpha):
    # normal: (3, H, W), alpha: (H, W)
    fg_mask = (alpha[None,...]>0.).repeat(3, 1, 1)
    normal = torch.where(fg_mask, torch.nn.functional.normalize(normal, p=2, dim=0), normal)

def rendered_world2cam(viewpoint_cam, normal, alpha, bg_color):
    # normal: (3, H, W), alpha: (H, W), bg_color: (3)
    # normal_cam: (3, H, W)
    _, H, W = normal.shape
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()
    normal_world = normal.permute(1,2,0).reshape(-1, 3) # (HxW, 3)
    normal_cam = torch.cat([normal_world, torch.ones_like(normal_world[...,0:1])], axis=-1) @ torch.inverse(torch.inverse(extrinsic_matrix).transpose(0,1))[...,:3]
    normal_cam = normal_cam.reshape(H, W, 3).permute(2,0,1) # (H, W, 3)
    if alpha != None:
        background = bg_color[...,None,None]
        normal_cam = normal_cam*alpha[None,...] + background*(1. - alpha[None,...])

    return normal_cam
def render_lighting(pc : GaussianModel, resolution=(512, 1024), sampled_index=None):
    if pc.brdf_mode=="envmap":
        lighting = extract_env_map(pc.brdf_mlp, resolution) # (H, W, 3)
        lighting = lighting.permute(2,0,1) # (3, H, W)
    else:
        raise NotImplementedError

    return lighting

def render_normal(viewpoint_camera, pc, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        # debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # if torch.is_tensor(d_xyz) is False:
    #     means3D = pc.get_xyz
    # else:
    #     means3D = from_homogenous(
    #         torch.bmm(d_xyz, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1))
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    out_extras = {}
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True) # (N, 3)
    normal_axis = pc.get_minimum_axis
    normal_axis, _ = flip_align_view(normal_axis, dir_pp_normalized)
    
    
    if colors_precomp is None:
        if True:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    if pipe.brdf:
        normal, delta_normal = pc.get_normal(dir_pp_normalized=dir_pp_normalized, return_delta=True) # (N, 3) 
        delta_normal = delta_normal.norm(dim=1, keepdim=True)


    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    p_hom = torch.cat([pc.get_xyz, torch.ones_like(pc.get_xyz[...,:1])], -1).unsqueeze(-1)
    p_view = torch.matmul(viewpoint_camera.world_view_transform.transpose(0,1), p_hom)
    p_view = p_view[...,:3,:]
    depth = p_view.squeeze()[...,2:3]
    depth = depth.repeat(1,3)

    render_extras = {}
    render_extras['depth'] = depth

    render_extras['normal_axis'] = 0.5*(normal_axis.reshape(-1,3)) + 0.5
    if pipe.brdf:
        
        render_extras['normal'] = 0.5*(normal.reshape(-1,3)) + 0.5
        
        render_extras['delta_normal'] = delta_normal.repeat(1, 3)

    out_extras = {}
    for k in render_extras.keys():
        if render_extras[k] is None: continue
        image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = render_extras[k],
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)[0]
        out_extras[k] = image
    
    for k in["normal", "normal_axis"]:
            if k in out_extras.keys():
                out_extras[k] = (out_extras[k] - 0.5) * 2. # range (0, 1) -> (-1, 1)

    
    raster_settings_alpha = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor([0,0,0], dtype=torch.float32, device="cuda"),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            # debug=False
        )
    rasterizer_alpha = GaussianRasterizer(raster_settings=raster_settings_alpha)
    alpha = torch.ones_like(means3D) 
    out_extras["alpha"] =  rasterizer_alpha(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = alpha,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)[0]
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    
    
    out_extras["normal_ref"] = render_normal_ref(viewpoint_cam=viewpoint_camera, depth=out_extras['depth'][0],bg_color=bg_color, alpha=out_extras['alpha'][0])
    out_extras["normal_ref_cam"] = rendered_world2cam(viewpoint_camera, out_extras["normal_ref"], out_extras['alpha'][0], bg_color)
    out_extras["normal_axis_cam"] = rendered_world2cam(viewpoint_camera, out_extras["normal_axis"], out_extras['alpha'][0], bg_color)
    
    if pipe.brdf:
        normalize_normal_inplace(out_extras["normal"], out_extras["alpha"][0])
            
        out_extras["normal_cam"] = rendered_world2cam(viewpoint_camera, out_extras["normal"], out_extras['alpha'][0], bg_color)
        out_extras['normal_uvmap'] = render_extras['normal']
        
    out = {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }
    out.update(out_extras)
    return out


def render_color(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        # debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # if torch.is_tensor(d_xyz) is False:
    #     means3D = pc.get_xyz
    # else:
    #     means3D = from_homogenous(
    #         torch.bmm(d_xyz, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1))
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    out_extras = {}
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True) # (N, 3)
    normal_axis = pc.get_minimum_axis
    normal_axis, _ = flip_align_view(normal_axis, dir_pp_normalized)
    
    
    if colors_precomp is None:
        if pipe.brdf:
            color_delta = None
            delta_normal = None
            
            gb_pos = pc.get_xyz # (N, 3) 
            view_pos = viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1) # (N, 3) 

            diffuse   = pc.get_diffuse # (N, 3) 
            normal, delta_normal = pc.get_normal(dir_pp_normalized=dir_pp_normalized, return_delta=True) # (N, 3) 
            delta_normal = delta_normal.norm(dim=1, keepdim=True)
            specular  = pc.get_specular # (N, 3) 
            roughness = pc.get_roughness # (N, 1) 
            color, brdf_pkg = pc.brdf_mlp.shade(gb_pos[None, None, ...], normal[None, None, ...], diffuse[None, None, ...], specular[None, None, ...], roughness[None, None, ...], view_pos[None, None, ...])

            colors_precomp = color.squeeze() # (N, 3) 
            diffuse_color = brdf_pkg['diffuse'].squeeze() # (N, 3) 
            specular_color = brdf_pkg['specular'].squeeze() # (N, 3) 

            if pc.brdf_dim>0:
                shs_view = pc.get_brdf_features.view(-1, 3, (pc.brdf_dim+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.brdf_dim, shs_view, dir_pp_normalized)
                color_delta = sh2rgb
                colors_precomp += color_delta

            

        elif pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if pipe.brdf:
        normal, delta_normal = pc.get_normal(dir_pp_normalized=dir_pp_normalized, return_delta=True) # (N, 3) 
        delta_normal = delta_normal.norm(dim=1, keepdim=True)


    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    p_hom = torch.cat([pc.get_xyz, torch.ones_like(pc.get_xyz[...,:1])], -1).unsqueeze(-1)
    p_view = torch.matmul(viewpoint_camera.world_view_transform.transpose(0,1), p_hom)
    p_view = p_view[...,:3,:]
    depth = p_view.squeeze()[...,2:3]
    depth = depth.repeat(1,3)

    render_extras = {}
    render_extras['depth'] = depth

    render_extras['normal_axis'] = 0.5*(normal_axis.reshape(-1,3)) + 0.5
    if pipe.brdf:
        render_extras['normal'] = 0.5*(normal.reshape(-1,3)) + 0.5
        render_extras['delta_normal'] = delta_normal.repeat(1, 3)
        render_extras.update({
                    "diffuse": diffuse, 
                    "specular": specular, 
                    "roughness": roughness.repeat(1, 3), 
                    "diffuse_color": diffuse_color, 
                    "specular_color": specular_color, 
                    "color_delta": color_delta,  
                    })

    out_extras = {}
    for k in render_extras.keys():
        if render_extras[k] is None: continue
        image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = render_extras[k],
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)[0]
        out_extras[k] = image
    
    for k in["normal", "normal_axis"]:
            if k in out_extras.keys():
                out_extras[k] = (out_extras[k] - 0.5) * 2. # range (0, 1) -> (-1, 1)

    
    raster_settings_alpha = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor([0,0,0], dtype=torch.float32, device="cuda"),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            # debug=False
        )
    rasterizer_alpha = GaussianRasterizer(raster_settings=raster_settings_alpha)
    alpha = torch.ones_like(means3D) 
    out_extras["alpha"] =  rasterizer_alpha(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = alpha,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)[0]
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    
    
    out_extras["normal_ref"] = render_normal_ref(viewpoint_cam=viewpoint_camera, depth=out_extras['depth'][0],bg_color=bg_color, alpha=out_extras['alpha'][0])
    out_extras["normal_ref_cam"] = rendered_world2cam(viewpoint_camera, out_extras["normal_ref"], out_extras['alpha'][0], bg_color)
    out_extras["normal_axis_cam"] = rendered_world2cam(viewpoint_camera, out_extras["normal_axis"], out_extras['alpha'][0], bg_color)

    if pipe.brdf:
        normalize_normal_inplace(out_extras["normal"], out_extras["alpha"][0])
            
        out_extras["normal_cam"] = rendered_world2cam(viewpoint_camera, out_extras["normal"], out_extras['alpha'][0], bg_color)
    
    out = {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }
    out.update(out_extras)
    return out