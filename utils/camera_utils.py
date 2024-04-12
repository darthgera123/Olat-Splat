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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch,NPtoTorch
from utils.graphics_utils import fov2focal
from PIL import Image

WARNED = False
def pil_or_np(input):
    if isinstance(input, Image.Image):
        # Input is a PIL image
        return 'PIL'
    elif isinstance(input, np.ndarray):
        # Input is a NumPy array
        return 'Numpy'
    else:
        return False
def loadCam(args, id, cam_info, resolution_scale):
    # img_type = pil_or_np(cam_info.image)
    # if img_type == 'PIL':
    #     orig_w, orig_h = cam_info.image.size
    # elif img_type == 'Numpy':
    #     orig_w, orig_h = cam_info.image.shape[0],cam_info.image.shape[1]

    # if args.resolution in [1, 2, 4, 8]:
    #     resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    # else:  # should be a type that converts to float
    #     if args.resolution == -1:
    #         if orig_w > 1600:
    #             global WARNED
    #             if not WARNED:
    #                 print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
    #                     "If this is not desired, please explicitly specify '--resolution/-r' as 1")
    #                 WARNED = True
    #             global_down = orig_w / 1600
    #         else:
    #             global_down = 1
    #     else:
    #         global_down = orig_w / args.resolution

    #     scale = float(global_down) * float(resolution_scale)
    #     resolution = (int(orig_w / scale), int(orig_h / scale))

    if cam_info.image is None:
        gt_image = None
        loaded_mask = None

        
    else:
        img_type = pil_or_np(cam_info.image)
        resolution=1
        if img_type == 'PIL':
            resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        elif img_type == 'Numpy':
            resized_image_rgb = NPtoTorch(cam_info.image, resolution)

        gt_image = resized_image_rgb[:3, ...]
        loaded_mask = None

        if resized_image_rgb.shape[1] == 4:
            loaded_mask = resized_image_rgb[3:4, ...]
    
    try:
        light_dir = cam_info.light_dir
    except AttributeError:
        light_dir = None

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,Cx = cam_info.Cx,Cy=cam_info.Cy, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device='cuda',light_dir=light_dir,
                  width=cam_info.width,height=cam_info.height)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
