from sys import argv
from argparse import ArgumentParser
import os
import numpy as np
from tqdm import tqdm
import json
import cv2

from imageio.v2 import imread,imwrite
import math


def load_json(json_path):
    with open(json_path, 'r') as h:
        data = json.load(h)
    return data  

def save_json(data,json_path):
    with open(json_path,'w') as file:
        json.dump(data, file, sort_keys=True, indent=4)

def parse_args():
    parser =  ArgumentParser(description="convert calib file to nerf format transforms.json")
    parser.add_argument("--cam_json", default="", help="specify calib file location")
    parser.add_argument("--light_json", default="", help="specify calib file location")
    parser.add_argument("--output", default="transforms.json", help="output path")
    parser.add_argument("--num_lights",default=100,type=int)
    parser.add_argument("--img_path", default="images/", help="location of folder with images")
    parser.add_argument("--mask_path", default="images/", help="location of folder with masks")
    parser.add_argument("--create_alpha",action='store_true', help="create_images")
    parser.add_argument("--ext", default="exr", help="extention of input images")
    parser.add_argument("--scale", default=1,type=int, help="scale")
    args = parser.parse_args()
    return args

def create_alpha(img_path,mask_path,alpha_path,scale,num_lights,num_cameras,ext='exr'):
	# assuming all images and mask are following same naming convention. Breaks if it doesnt
    images = sorted(os.listdir(img_path))
    N = num_cameras

    for i in tqdm(range(0,N)):
        mask = cv2.imread(os.path.join(mask_path,f'Cam_{str(i).zfill(2)}.jpg'),cv2.IMREAD_GRAYSCALE)
        h,w = mask.shape
        nh,nw = int(h/scale),int(w/scale)
        for l in range(1,num_lights+1):
            if ext == 'exr':
                img = imread(os.path.join(img_path,f'L_{str(l).zfill(3)}/fg/Cam_{str(i).zfill(2)}.exr'))
                mask[mask<100] = 0
                nimg = cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA)
                nmask = cv2.resize(mask,(nw,nh),interpolation=cv2.INTER_AREA)/255.0
                alpha = nimg[...,:3] * (nmask)[...,np.newaxis]
                imwrite(os.path.join(alpha_path,f'Cam{str(i).zfill(2)}_L{str(l).zfill(3)}.exr'),alpha.astype('float32'))
                png_alpha = np.clip(np.power(np.clip(alpha,0,None)+1e-5,0.45),0,1)*255 
                imwrite(os.path.join(alpha_path,f'Cam{str(i).zfill(2)}_L{str(l).zfill(3)}.png'),png_alpha.astype('uint8'))


            elif ext == 'jpg':
                img = cv2.imread(os.path.join(img_path,f'L_{str(l).zfill(3)}/fg/Cam_{str(i).zfill(2)}.jpg'))
                nimg = cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA)
                nmask = cv2.resize(mask,(nw,nh),interpolation=cv2.INTER_AREA)  
                alpha = cv2.merge((nimg,nmask))
                cv2.imwrite(os.path.join(alpha_path,f'Cam{str(i).zfill(2)}_L{str(l).zfill(3)}.png'),alpha.astype('uint8'))



if __name__ == '__main__':
    
    args = parse_args()
    os.makedirs(args.output,exist_ok=True)
    cameras = load_json(args.cam_json)
    lights = load_json(args.light_json)
    n_cameras = len(cameras["frames"])

    all_lights = list(range(0,args.num_lights))
    all_cams = list(range(0,n_cameras))
    
    img_folder = os.path.join(args.output,'images/')
    if args.create_alpha:
        os.makedirs(img_folder,exist_ok=True)
        create_alpha(args.img_path,args.mask_path,img_folder,\
                     args.scale,args.num_lights,n_cameras,args.ext)
    
    
    
    
    # single olat
    test_cams = [5]
    test_lights = [5,28]
    train_lights = [l for l in all_lights if l not in test_lights]
    train_cams = [c for c in all_cams if c not in test_cams]
    
    transforms = dict()
    frames = []
    transforms["aabb_scale"] = cameras["aabb_scale"] 
    for l in train_lights:
        for c in train_cams:
            frame = dict()
            frame = cameras['frames'][c].copy()
            frame['light_direction'] = lights['light_dir'][l]
            frame['file_path'] = os.path.join(img_folder,f'Cam{str(c).zfill(2)}_L{str(l+1).zfill(3)}')
            frames.append(frame)
    print(train_lights)
    print(len(frames))
    transforms["frames"] = frames
    save_json(transforms,os.path.join(args.output,'transforms_train.json'))

    transforms = dict()
    frames = []
    transforms["aabb_scale"] = cameras["aabb_scale"] 
    for l in train_lights:
        for c in test_cams:
            frame = dict()
            frame = cameras['frames'][c].copy()
            frame['light_direction'] = lights['light_dir'][l]
            frame['file_path'] = os.path.join(img_folder,f'Cam{str(c).zfill(2)}_L{str(l+1).zfill(3)}')
            frames.append(frame)
    if len(test_lights) > 0:
        for l in test_lights:
            for c in train_cams:
                frame = dict()
                frame = cameras['frames'][c].copy()
                frame['light_direction'] = lights['light_dir'][l]
                frame['file_path'] = os.path.join(img_folder,f'Cam{str(c).zfill(2)}_L{str(l+1).zfill(3)}')
                frames.append(frame)
        for l in test_lights:
            for c in test_cams:
                frame = dict()
                frame = cameras['frames'][c].copy()
                frame['light_direction'] = lights['light_dir'][l]
                frame['file_path'] = os.path.join(img_folder,f'Cam{str(c).zfill(2)}_L{str(l+1).zfill(3)}')
                frames.append(frame)
    
    
    # print(test_lights)
    print(len(frames))
    transforms["frames"] = frames
    save_json(transforms,os.path.join(args.output,'transforms_test.json'))

