from sys import argv
from argparse import ArgumentParser
import os
import numpy as np
from tqdm import tqdm
import json
import cv2

from imageio.v2 import imread,imwrite
import math
import open3d as o3d
from plyfile import PlyData, PlyElement
from create_test_pose import generate_interpolated_poses
import copy
import subprocess

def parse_args():
    parser =  ArgumentParser(description="convert calib file to nerf format transforms.json")
    parser.add_argument("--calib", default="", help="specify calib file location")
    parser.add_argument("--output", default="transforms.json", help="output path")
    parser.add_argument("--create_alpha",action='store_true', help="create_images")
    parser.add_argument("--create_points",action='store_true', help="create_point_cloud")
    parser.add_argument("--create_lights",action='store_true', help="create_lights")
    parser.add_argument("--create_spiral",action='store_true', help="create_lights")
    parser.add_argument("--points_in", default="points3d.ply", help="location of folder with images")
    parser.add_argument("--obj", default="points3d.ply", help="loocation of obj_file")
    parser.add_argument("--imh", default=1440,type=int, help="height of image")
    parser.add_argument("--imw", default=810,type=int, help="width of image")
    parser.add_argument("--img_path", default="images/", help="location of folder with images")
    parser.add_argument("--mask_path", default="images/", help="location of folder with masks")
    parser.add_argument("--ext", default="exr", help="extention of input images")
    parser.add_argument("--lights_txt",
                        default="/CT/prithvi/work/studio-tools/LightStage/calibration/LSX_light_positions_aligned.txt",help="path to light dir (mm)")
    parser.add_argument("--lights_order", 
                        default="/CT/prithvi2/work/lightstage/LSX_python3_command/LSX/Data/LSX3_light_z_spiral.txt", help="order of lights")
    parser.add_argument("--envmap",
                        default="red_bedroom",help="path to light dir (mm)")
    parser.add_argument("--scale", default=1,type=int, help="scale")
    args = parser.parse_args()
    return args

def read_light_dir(filename,scale=1):                          
     numbers_list = []                                          
     with open(filename, 'r') as file:                          
             for line in file:                          
                     numbers = line.strip().split()     
                     numbers = [float(n)/scale for n in numbers]        
                     numbers_list.append(numbers)          
     return numbers_list

def light_order(filename,lights):
    with open(filename) as light_file:
        z_spiral = np.asarray([int(line.strip())-1 for line in light_file if line.strip()]) 
    
    return np.asarray(lights)[z_spiral].tolist()
          


def read_calib(calib_dir):
    INTR = []
    EXTR = []   
    with open(calib_dir, 'r') as fp:
        while True:
            text = fp.readline()
            if not text:
                break

            elif "extrinsic" in text:
                lines = []
                for i in range(3):
                    line = fp.readline()
                    lines.append([float(w) for w in line.strip('#').strip().split()])
                lines.append([0, 0, 0, 1])
                extrinsic = np.array(lines).astype(np.float32)
                EXTR.append(np.linalg.inv(extrinsic))

            elif "intrinsic" in text:
                lines = []
                line = fp.readline()
                if "time" in line:
                    continue
                for i in range(3):
                    lines_3 = [float(w) for w in line.strip('#').strip().split()]
                    lines.append(lines_3)
                    line = fp.readline()
                intrinsic = np.array(lines).astype(np.float32)
                INTR.append(intrinsic)
               
    return INTR,EXTR    

def load_json(json_path):
    with open(json_path, 'r') as h:
        data = json.load(h)
    return data  

def save_json(data,json_path):
    with open(json_path,'w') as file:
        json.dump(data, file, sort_keys=True, indent=4)

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom


def central_point(out,transform_matrix_pcl):
    # find a central point they are all looking at
    print("computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in out["frames"]:
        mf = np.array(f["transform_matrix"])[0:3,:]
        for g in out["frames"]:
            mg = g["transform_matrix"][0:3,:]
            p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            if w > 0.0001:
                totp += p*w
                totw += w
    totp /= totw
    print(totp) # the cameras are looking at totp
    for f in out["frames"]:
        f["transform_matrix"][0:3,3] -= totp
        f["transform_matrix"] = f["transform_matrix"].tolist()
    center_origin = np.asarray([0.5,0.1,0.5])
	# transform_matrix_pcl[0:3, 3] -= center_origin[0:3]
    transform_matrix_pcl[0:3, 3] -= totp[0:3]
    return out,transform_matrix_pcl,totp


def create_alpha(img_path,mask_path,alpha_path,scale,ext='exr'):
	# assuming all images and mask are following same naming convention. Breaks if it doesnt
    images = sorted(os.listdir(img_path))
    masks = sorted(os.listdir(mask_path))
    N = len(masks)
    for i in tqdm(range(0,N)):
        mask = cv2.imread(os.path.join(mask_path,f'Cam_{str(i).zfill(2)}.jpg'),cv2.IMREAD_GRAYSCALE)
        h,w = mask.shape
        nh,nw = int(h/scale),int(w/scale)
        if ext == 'exr':
            img = imread(os.path.join(img_path,f'Cam_{str(i).zfill(2)}.exr'))
            mask[mask<100] = 0
            nimg = cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA)
            nmask = cv2.resize(mask,(nw,nh),interpolation=cv2.INTER_AREA)/255.0
            alpha = nimg[...,:3] * (nmask)[...,np.newaxis]
            alpha = np.clip(alpha,0,None)
            imwrite(os.path.join(alpha_path,f'Cam_{str(i).zfill(2)}.exr'),alpha.astype('float32'))
            png_alpha = np.clip(np.power(np.clip(alpha,0,None)+1e-5,0.45),0,1)*255 
            imwrite(os.path.join(alpha_path,f'Cam_{str(i).zfill(2)}.png'),png_alpha.astype('uint8'))


        elif ext == 'jpg':
            img = cv2.imread(os.path.join(img_path,f'Cam_{str(i).zfill(2)}.jpg'))
            nimg = cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA)
            nmask = cv2.resize(mask,(nw,nh),interpolation=cv2.INTER_AREA)  
            alpha = cv2.merge((nimg,nmask))
            cv2.imwrite(os.path.join(alpha_path,f'Cam_{str(i).zfill(2)}.png'),alpha.astype('uint8'))
        
     
		

def parse_cameras(calib,path,mask_path,imw,imh):
    intr,extr = read_calib(calib)
    transforms = {}
    transforms["aabb_scale"] = 16.0
    frames = []
    for idx in range(len(intr)):
        frame = {}
        frame['w'] = imw
        frame['h'] = imh
        frame['fl_x'] = np.asarray(intr[idx])[0,0] * imw
        frame['fl_y'] = np.asarray(intr[idx])[1,1] * imw
        frame['cx'] = np.asarray(intr[idx])[0,2] * imw
        frame['cy'] = np.asarray(intr[idx])[1,2] * imw
        frame['camera_angle_x'] = math.atan(float(imw) / (float(frame['fl_x'] * 2))) * 2
        frame['camera_angle_y'] = math.atan(float(imh) / (float(frame['fl_y'] * 2))) * 2
        frame['transform_matrix'] = extr[idx][[2,0,1,3],:]
        # frame['transform_matrix'] = extr[idx]
        frame['file_path'] = os.path.join(path,f'Cam_{str(idx).zfill(2)}')
        frame['mask_path'] = os.path.join(mask_path,f'Cam_{str(idx).zfill(2)}.jpg')
        frames.append(frame)
    transforms["frames"] = frames
    return transforms

def gentritex(v, vt, vi, vti, texsize):
    """Create 3 texture maps containing the vertex indices, texture vertex
    indices, and barycentric coordinates"""
    vt = np.array(vt, dtype=np.float32)
    vi = np.array(vi, dtype=np.int32)
    vti = np.array(vti, dtype=np.int32)
    ntris = vi.shape[0]

    texu, texv = np.meshgrid(
            (np.arange(texsize) + 0.5) / texsize,
            (np.arange(texsize) + 0.5) / texsize)
    texuv = np.stack((texu, texv), axis=-1)

    vt = vt[vti]

    viim = np.zeros((texsize, texsize, 3), dtype=np.int32)
    vtiim = np.zeros((texsize, texsize, 3), dtype=np.int32)
    baryim = np.zeros((texsize, texsize, 3), dtype=np.float32)

    for i in list(range(ntris))[::-1]:
        bbox = (
            max(0, int(min(vt[i, 0, 0], min(vt[i, 1, 0], vt[i, 2, 0]))*texsize)-1),
            min(texsize, int(max(vt[i, 0, 0], max(vt[i, 1, 0], vt[i, 2, 0]))*texsize)+2),
            max(0, int(min(vt[i, 0, 1], min(vt[i, 1, 1], vt[i, 2, 1]))*texsize)-1),
            min(texsize, int(max(vt[i, 0, 1], max(vt[i, 1, 1], vt[i, 2, 1]))*texsize)+2))
        v0 = vt[None, None, i, 1, :] - vt[None, None, i, 0, :]
        v1 = vt[None, None, i, 2, :] - vt[None, None, i, 0, :]
        v2 = texuv[bbox[2]:bbox[3], bbox[0]:bbox[1], :] - vt[None, None, i, 0, :]
        d00 = np.sum(v0 * v0, axis=-1)
        d01 = np.sum(v0 * v1, axis=-1)
        d11 = np.sum(v1 * v1, axis=-1)
        d20 = np.sum(v2 * v0, axis=-1)
        d21 = np.sum(v2 * v1, axis=-1)
        denom = d00 * d11 - d01 * d01

        if denom != 0.:
            baryv = (d11 * d20 - d01 * d21) / denom
            baryw = (d00 * d21 - d01 * d20) / denom
            baryu = 1. - baryv - baryw

            baryim[bbox[2]:bbox[3], bbox[0]:bbox[1], :] = np.where(
                    ((baryu >= 0.) & (baryv >= 0.) & (baryw >= 0.))[:, :, None],
                    np.stack((baryu, baryv, baryw), axis=-1),
                    baryim[bbox[2]:bbox[3], bbox[0]:bbox[1], :])
            viim[bbox[2]:bbox[3], bbox[0]:bbox[1], :] = np.where(
                    ((baryu >= 0.) & (baryv >= 0.) & (baryw >= 0.))[:, :, None],
                    np.stack((vi[i, 0], vi[i, 1], vi[i, 2]), axis=-1),
                    viim[bbox[2]:bbox[3], bbox[0]:bbox[1], :])
            vtiim[bbox[2]:bbox[3], bbox[0]:bbox[1], :] = np.where(
                    ((baryu >= 0.) & (baryv >= 0.) & (baryw >= 0.))[:, :, None],
                    np.stack((vti[i, 0], vti[i, 1], vti[i, 2]), axis=-1),
                    vtiim[bbox[2]:bbox[3], bbox[0]:bbox[1], :])

    return viim, vtiim, baryim

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def read_obj(filename):
    v = []
    vt = []
    vn = []
    vindices = []
    vtindices = []

    with open(filename, "r") as f:
        while True:
            line = f.readline()

            if line == "":
                break

            if line[:2] == "v ":
                v.append([float(x) for x in line.split()[1:]])
            elif line[:2] == "vt":
                vt.append([float(x) for x in line.split()[1:]])
            elif line[:2] == "f ":
                vindices.append([int(entry.split('/')[0]) - 1 for entry in line.split()[1:]])
                if line.find("/") != -1:
                    vtindices.append([int(entry.split('/')[1]) - 1 for entry in line.split()[1:]])
    
    return v,vt,vindices,vtindices

def mask_and_sample_points(vt, tidxim, colors, stridey, stridex):
    """
    Masks out empty regions in a texture map and samples points from non-empty regions.

    Parameters:
    - vt (numpy.ndarray): Texture vertex positions.
    - tidxim (numpy.ndarray): Texture vertex indices.
    - colors (numpy.ndarray): Colors corresponding to vertices.
    - stridey (int): Vertical stride for sampling points.
    - stridex (int): Horizontal stride for sampling points.
    
    Returns:
    - sampled_points (numpy.ndarray): Sampled points from non-empty regions.
    - sampled_colors (numpy.ndarray): Colors corresponding to sampled points.
    """

    # Convert tidxim to integer type to use as indices
    tidxim = tidxim.astype(int)

    # Convert VT array to integer type to use as indices
    vt = np.asarray(vt).astype(int)

    # Initialize arrays to store sampled points and colors
    sampled_points = []
    sampled_colors = []

    # Iterate through tidxim to sample points
    for y in range(0, tidxim.shape[0], stridey):
        for x in range(0, tidxim.shape[1], stridex):
            # Get the texture vertex index
            texture_vertex_index = tidxim[y, x, 0]
            
            # Check if texture vertex index is valid
            if 0 <= texture_vertex_index < len(colors):
                # Use texture vertex index to get UV coordinates
                uv_coords = vt[texture_vertex_index]
                
                # If UV coordinates are valid, append them to sampled points and corresponding color
                if 0 <= uv_coords[0] < vt.shape[0] and 0 <= uv_coords[1] < vt.shape[1]:
                    sampled_points.append((y, x))
                    sampled_colors.append(colors[texture_vertex_index])
    # Convert lists to numpy arrays
    sampled_points = np.array(sampled_points)
    sampled_colors = np.array(sampled_colors)

    return sampled_points, sampled_colors



if __name__ == '__main__':
    
    args = parse_args()
    os.makedirs(args.output,exist_ok=True)
    img_folder = os.path.join(args.output,'images/')
    if args.create_alpha:
        os.makedirs(img_folder,exist_ok=True)
        create_alpha(args.img_path,args.mask_path,img_folder,args.scale,args.ext)

    sensor_x,sensor_y = 10.0000, 17.777
    transforms = parse_cameras(args.calib,img_folder,args.mask_path,args.imw,args.imh)
    
    
    
    transform_matrix_pcl = np.eye(4,dtype='float32')
    # Rotation of axis
    transform_matrix_pcl = transform_matrix_pcl[[2,0,1,3],:]
    transforms,transforms_pcd,center = central_point(transforms,transform_matrix_pcl)
    
    if args.create_points:
        v,vt,vindices,vtindices = read_obj(args.obj)
        texsize = 1024
    
        idxim,tidxim,barim = gentritex(v,vt,vindices,vtindices,texsize)
        # print(idxim.shape)
        geo=np.array(v)[:,:3]
        colors = np.array(v)[:,3:]
        # Convert to homogeneous coordinates by adding a fourth component (w = 1)
        homogeneous_vertices = np.hstack([geo, np.ones((geo.shape[0], 1))])
        transformed_vertices = homogeneous_vertices.dot(transforms_pcd.T)
        # Convert back to 3D coordinates
        # If the w component is not 1, you need to divide by it
        geo = transformed_vertices[:, :3] / transformed_vertices[:, 3, np.newaxis] #(N,3)
        
        
        sample_x,sample_y = 256,256
        uvheight, uvwidth = texsize,texsize #4096x4096
        stridey = uvheight // sample_x #sampling frequenc will set points
        stridex = uvwidth // sample_y

        # sampled_points, sampled_colors = mask_and_sample_points(vt, tidxim, colors, stridey, stridex)

        v0 = geo[idxim[stridey//2::stridey, stridex//2::stridex, 0], :]
        v1 = geo[idxim[stridey//2::stridey, stridex//2::stridex, 1], :]
        v2 = geo[idxim[stridey//2::stridey, stridex//2::stridex, 2], :]
        

        # vt0 = vt[tidxim[stridey//2::stridey, stridex//2::stridex, 0], :]
        # vt1 = vt[tidxim[stridey//2::stridey, stridex//2::stridex, 1], :]
        # vt2 = vt[tidxim[stridey//2::stridey, stridex//2::stridex, 2], :]

        c0 = colors[idxim[stridey//2::stridey, stridex//2::stridex, 0], :]
        c1 = colors[idxim[stridey//2::stridey, stridex//2::stridex, 1], :]
        c2 = colors[idxim[stridey//2::stridey, stridex//2::stridex, 2], :]
        
        ply_vertex = (
                        barim[None,stridey//2::stridey, stridex//2::stridex, 0, None] * v0 +
                        barim[None,stridey//2::stridey, stridex//2::stridex, 1, None] * v1 +
                        barim[None,stridey//2::stridey, stridex//2::stridex, 2, None] * v2
                        )
        

        
        ply_colors = (
                        barim[None,stridey//2::stridey, stridex//2::stridex, 0, None] * c0 +
                        barim[None,stridey//2::stridey, stridex//2::stridex, 1, None] * c1 +
                        barim[None,stridey//2::stridey, stridex//2::stridex, 2, None] * c2
                        )
        
        ply_vertex=ply_vertex.reshape(sample_x*sample_y,3)
        ply_colors=ply_colors.reshape(sample_x*sample_y,3)
        # ply_vertex = np.zeros((sampled_points.shape[0], 3))
        # ply_colors = np.zeros((sampled_colors.shape[0], 3))

        # for i, (y, x) in enumerate(sampled_points):
        #     uv_index = tidxim[y // stridey, x // stridex, 0]
        #     uv_coords = vt[uv_index]
        #     ply_vertex[i] = barim[y // stridey, x // stridex, 0] * geo[idxim[y // stridey, x // stridex, 0]] + \
        #                     barim[y // stridey, x // stridex, 1] * geo[idxim[y // stridey, x // stridex, 1]] + \
        #                     barim[y // stridey, x // stridex, 2] * geo[idxim[y // stridey, x // stridex, 2]]
        #     ply_colors[i] = sampled_colors[i]


        ply_save = os.path.join(args.output,'points3d.ply')
        img_save = os.path.join(args.output,'texture.png')
        imwrite(img_save,(ply_colors.reshape(sample_x,sample_y,3)*255).astype('uint8'))
        storePly(ply_save,ply_vertex,(ply_colors*255).astype('uint8'))
        # some spurious points also coming clean manually?
    else:
        pcd = o3d.io.read_point_cloud(args.points_in)
        pcd.transform(transforms_pcd)
        point_cloud_out = os.path.join(args.output,'points3d.ply')
        o3d.io.write_point_cloud(point_cloud_out, pcd)
    
    # envmap_names = ["9C4A0003-e05009bcad", "9C4A0004-db1d4a14f3", "9C4A0006-5133111e97" ,\
    #                     "9C4A0010-48093a025c", "9C4A0022-6d8fe2e88e", "9C4A0027-9a856d679a", \
    #                     "9C4A0029-372e11316d", "9C4A0034-a460e29cd9", "9C4A0037-b2d1efd096", \
    #                     "9C4A0045-76d25bb135", "9C4A0046-6f317ef205", "9C4A0048-48d7dfa6b0", \
    #                     "9C4A0052-9b4fa1e4a1", "9C4A0064-2bf2b7e178", "9C4A0069-8ccc2075f2", \
    #                     "9C4A0071-20e04984d0", "9C4A0076-46c51ec2c2", "9C4A0079-7fb521e807", \
    #                     "9C4A0087-5ad3395167", "9C4A0088-5162e36ae6", "9C4A0090-81b4c61bc1", \
    #                     "9C4A0094-c96fec6e86", "9C4A0106-3656c02faf", "9C4A0120-0fd27f2a38", \
    #                     "9C4A0121-ff0fd23bdf", "9C4A0129-34b0d8cec2", "9C4A0132-07352d1dd0", \
    #                     "9C4A0136-05a3e2b220", "9C4A0137-73c0813158", "9C4A0148-c51c9e3c93"]

    # train_env_maps = [  "9C4A0004-db1d4a14f3","9C4A0027-9a856d679a","9C4A0045-76d25bb135",\
    #                     "9C4A0048-48d7dfa6b0","9C4A0069-8ccc2075f2","9C4A0071-20e04984d0",\
    #                     "9C4A0076-46c51ec2c2", "9C4A0079-7fb521e807","9C4A0106-3656c02faf",\
    #                     "9C4A0137-73c0813158"  ]
    # test_env_maps = ["9C4A0003-e05009bcad","9C4A0120-0fd27f2a38"]
        
    with open('envmap.txt', 'r') as file:
    # Read all lines from the file and store them in a list
        lines = [line.strip() for line in file]
    train_env_maps = lines[0:500]
    test_env_maps = lines[0:10] + lines[1000:1002]
    
    # if args.create_lights:
    #     lights = {}
    #     usc_lights = np.asarray(read_light_dir(args.lights_txt,scale=1000))[:,[2,0,1]]
    #     lights['light_dir'] = [(direction - center).tolist() for direction in usc_lights]
    #     lights['light_dir'] = light_order(args.lights_order,lights['light_dir'])
    #     save_json(lights,os.path.join(args.output,'light_dir.json'))
    if args.create_lights:
        # envmap = os.path.join('/CT/LS_BRM03/nobackup/indoor_envmap/indoor_scene',args.envmap)

        # transforms['envmap'] = envmap + '.exr'
        
    
        if args.create_spiral:
            # N = 50
            N=1
            indices = [5,14,18,19,5]
            test_indices = [5,18,38]
            transforms_test = dict()
            transforms_test["aabb_scale"] = 16.0

            frames = []
            # for i in range(0,len(indices[:-1])):
            #     start_frame = transforms["frames"][indices[i]]
            #     pose1 = np.asarray(start_frame["transform_matrix"])
            #     pose2 = np.asarray(transforms["frames"][indices[i+1]]["transform_matrix"])
            #     poses = generate_interpolated_poses(pose1,pose2,N)
                
            #     for i in range(0,len(poses)):
            #         frame={}
            #         frame = start_frame.copy()
            #         frame["transform_matrix"] = poses[i].tolist()
            #         frames.append(frame)


            
            
            for ind in test_indices:
                for env in test_env_maps:
                    
                    frame_copy = copy.deepcopy(transforms['frames'][ind])
        
                    frame_copy['file_path'] = os.path.join(args.img_path, env, f'images/Cam_{str(ind).zfill(2)}')
                    frame_copy['envmap'] = os.path.join('/CT/LS_BRM03/nobackup/indoor_envmap/med_exr', env + '.exr')
                    frames.append(frame_copy)
                    
            transforms_test["frames"] = frames
            
            save_json(transforms_test,os.path.join(args.output,'transforms_test.json'))
            
            # all_frames = transforms["frames"]
            transforms_train = dict()
            transforms_test["aabb_scale"] = 16.0
            frames = []
            for ind in range(0,len(transforms['frames'])):
                if ind in test_indices:
                    continue
                for env in train_env_maps:
                    
                    frame_copy = copy.deepcopy(transforms['frames'][ind])
        
                    frame_copy['file_path'] = os.path.join(args.img_path, env, f'images/Cam_{str(ind).zfill(2)}')
                    frame_copy['envmap'] = os.path.join('/CT/LS_BRM03/nobackup/indoor_envmap/med_exr', env + '.exr')
                    frames.append(frame_copy)
            
            transforms_train["frames"] = frames
            
            save_json(transforms_train,os.path.join(args.output,'transforms_train.json'))
        else:
            save_json(transforms,os.path.join(args.output,'transforms_test.json'))
            save_json(transforms,os.path.join(args.output,'transforms_train.json'))
    
    
        
    