from sys import argv
from argparse import ArgumentParser
import os
import numpy as np
from tqdm import tqdm
import json
import cv2
import re
from imageio.v2 import imread,imwrite
import xml.etree.ElementTree as ET
import math
import open3d as o3d
import cv2
# Hardcoding certain things for now

def parse_args():
	parser =  ArgumentParser(description="convert Agisoft XML export to nerf format transforms.json")

	parser.add_argument("--camera", default="", help="specify xml file location")
	parser.add_argument("--input", default="", help="specify pcl file location")
	parser.add_argument("--output", default="transforms.json", help="output path")
	parser.add_argument("--points_out", default="points3d.ply", help="output path")
	# parser.add_argument("--create_alpha",action='store_true', help="create_images")
	parser.add_argument("--points_in", default="images/", help="location of folder with images")
	# parser.add_argument("--mask_path", default="images/", help="location of folder with images")
	# parser.add_argument("--imgtype", default="png", help="type of images (ex. jpg, png, ...)")
	# parser.add_argument("--scale", default=1,type=int, help="scale")
	args = parser.parse_args()
	return args


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
    return out,transform_matrix_pcl


if __name__ == '__main__':
    
    args = parse_args()
    file_path = args.camera  # Replace this with your file path
    imw,imh = 810,1440
    # transforms = load_json(args.input)
    transforms_neus2 = load_json(args.input)
    # camera_data = parse_camera_data(file_path)
    transforms = transforms_neus2
    pcd = o3d.io.read_point_cloud(args.points_in)
    scale = 4
    path = '/CT/prithvi2/static00/3dgs/data/bluesweater_exr/oleks_intri1/images/'


    for idx,frame in enumerate(transforms['frames']):
        frame["file_path"] = os.path.join(path,f'Cam_{str(idx).zfill(2)}')
        frame['transform_matrix'] = np.asarray(frame['transform_matrix'])[[2,0,1,3],:]
        frame['fl_x'] = np.asarray(frame['intrinsic_matrix'])[0,0]/scale
        frame['fl_y'] = np.asarray(frame['intrinsic_matrix'])[1,1]/scale
        frame['cx'] = np.asarray(frame['intrinsic_matrix'])[0,2]/scale
        frame['cy'] = np.asarray(frame['intrinsic_matrix'])[1,2]/scale
        frame['camera_angle_x'] = math.atan(float(imw) / (float(frame['fl_x'] * 2))) * 2
        frame['camera_angle_y'] = math.atan(float(imh) / (float(frame['fl_y'] * 2))) * 2
        frame['w'] = imw
        frame['h'] = imh
    print(math.degrees(frame['camera_angle_y']))
    # print(frame['cx'])
    transform_matrix_pcl = np.eye(4,dtype='float32')
    transform_matrix_pcl = transform_matrix_pcl[[2,0,1,3],:]
    transforms,transforms_pcd = central_point(transforms,transform_matrix_pcl)
    # pcd.scale(2.5, center=pcd.get_center())
    pcd.transform(transforms_pcd)
    
    save_json(transforms,os.path.join(args.output,'transforms_train.json'))
    # save_json(transforms,os.path.join(args.output,'transforms_calib.json'))
    point_cloud_out = os.path.join(args.output,'points3d.ply')
    o3d.io.write_point_cloud(point_cloud_out, pcd)    
    # print(extr[0])
    # with open(output_file, 'w') as outfile:
	# 	json.dump(formatted_data, outfile, indent=4)
