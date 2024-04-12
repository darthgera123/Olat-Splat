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

def parse_args():
    parser =  ArgumentParser(description="convert calib file to nerf format transforms.json")
    parser.add_argument("--input_json", default="", help="specify calib file location")
    
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


def central_point(out):
    # find a central point they are all looking at
    print("computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in out["frames"]:
        mf = np.array(f["transform_matrix"])[0:3,:]
        for g in out["frames"]:
            mg = np.array(g["transform_matrix"])[0:3,:]
            p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            if w > 0.0001:
                totp += p*w
                totw += w
    totp /= totw
    print(totp) # the cameras are looking at totp
    return totp

def update_center(out,center):
    for f in out["frames"]:
        g = np.asarray(f["transform_matrix"])
        g[0:3,3] -= center
        f["transform_matrix"] = g.tolist()
    return out

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
        

if __name__ == '__main__':
    
    args = parse_args()
   

    sensor_x,sensor_y = 10.0000, 17.777
    # transforms = parse_cameras(args.calib,img_folder,args.mask_path,args.imw,args.imh)
    transforms = load_json(args.input_json)
    center = central_point(transforms)
    transforms = update_center(transforms,center)
    center = central_point(transforms)
    # transforms,center = central_point(transforms)
    
    
   
    
    
        
    