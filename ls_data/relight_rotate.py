import cv2
import sys

from argparse import ArgumentParser
import argparse
from imageio.v3 import imread,imwrite

import os
import numpy as np
from tqdm import tqdm
# import exr2numpy

def parse_args():
    parser =  ArgumentParser(description="convert calib file to nerf format transforms.json")
    
    parser.add_argument("--input", default="", help="specify calib file location")
    parser.add_argument("--output_dir", default="transforms.json", help="output path")
    parser.add_argument("--map", default="", help="specify calib file location")
    parser.add_argument("--order", default="transforms.json", help="output path")
    parser.add_argument("--envmap", default="", help="specify calib file location")
    parser.add_argument("--envmap_name", default="", help="specify calib file location")
    parser.add_argument("--mask_path", default="", help="specify calib file location")
    parser.add_argument("--bg", default="", help="specify calib file location")
    parser.add_argument("--type", default="train", help="specify calib file location")
    args = parser.parse_args()
    return args
# Get an optical flow model. As as example, we will use RAFT Small
# with the weights pretrained on the FlyingThings3D dataset
def norm_img(image):
    min_value = 0.0
    max_value = 1.0

    image_min = np.min(image)
    image_max = np.max(image)
    
    # Normalize the image to the specified range
    img = (image - image_min) / (image_max - image_min) 
    return img

def load_image(path):
    # image = cv2.imread(path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.clip(imread(path),0,None)
    h,w,c = image.shape
    
    img = norm_img(image)
    return img
def read_numbers_from_file(file_path):
    numbers = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Convert each line to a number, assuming they are integers
                # Use float(line.strip()) if the numbers are floating-point
                numbers.append(int(line.strip()))
    except Exception as e:
        print(f"An error occurred: {e}")
    return numbers

if __name__ == '__main__':
    
    args = parse_args()
    full_lights = [1,15,35,56,77,98,119,140,161,182,203,224,245,266,287,308,329,350,363]
    lights = [x for x in range(15, 364) if x not in full_lights]
    canon = [203]
    bright = [363]
    canon_track = os.path.join(args.input,f'L_{str(canon[0]).zfill(3)}')
    mapy = np.asarray(read_numbers_from_file(args.map)).reshape(256,512)
    order = np.asarray(read_numbers_from_file(args.order))
    
    count = 0
    for pix in range(0,512,8):
        envmap_folder = args.envmap
        envmap_path = os.path.join(args.envmap,f'{args.envmap_name}_{pix}.exr')
        envmap = imread(envmap_path)
        envmap = norm_img(envmap)
        envmap_name = f'{args.envmap_name}'
    
    
    
        colors = []
        for i in tqdm(range(331)):
            mask = np.zeros((256,512,3)).astype('uint8')
            mask[mapy==(order[i]-1)] = 1
            
            masked = envmap*mask
            intensity_sum = np.sum(masked, axis=2)
            max_intensity_location = np.unravel_index(np.argmax(intensity_sum), intensity_sum.shape)
            color = masked[max_intensity_location]

            # color = np.mean(envmap[mask])
        
            colors.append(color)
        
        alpha = 1.0
        
        output_dir = os.path.join(args.output_dir,envmap_name)
        os.makedirs(output_dir,exist_ok=True)
        i = 0
        images = []
        alpha_path = os.path.join(output_dir,args.type)
        # clear_folder_content(alpha_path)
        os.makedirs(alpha_path,exist_ok=True)
                        
        for l,light in enumerate(tqdm(lights)):    
            light_dir = os.path.join(args.input,\
                        f'L_{str(light).zfill(3)}/{args.type}/ours_20000/renders/{str(i).zfill(5)}.exr')
            img = load_image(light_dir)
            images.append(colors[l]*img)
        relit = np.sum(images,axis=0)
        relit = norm_img(relit)

        del images 
        if args.mask_path != "":
            mask = cv2.imread(os.path.join(args.mask_path,f'{str(i).zfill(5)}.png'),cv2.IMREAD_GRAYSCALE)
            h,w = mask.shape
            nh,nw = int(h/1),int(w/1)
            mask[mask<100] = 0
            nmask = cv2.resize(mask,(nw,nh),interpolation=cv2.INTER_AREA)/255.0
            alpha = relit[...,:3] * (nmask)[...,np.newaxis]
            
            
            alpha = np.clip(alpha,0,None)
            imwrite(os.path.join(alpha_path,f'Cam_{str(i).zfill(2)}.exr'),alpha.astype('float32'))
            
            png_alpha = np.clip(np.power(np.clip(alpha,0,None)+1e-5,0.45),0,1)*255 
            imwrite(os.path.join(alpha_path,f'Cam_{str(i).zfill(2)}.png'),png_alpha.astype('uint8'))
            if args.bg != "":
                os.makedirs(os.path.join(alpha_path,"bg"),exist_ok=True)
                bg = imread(os.path.join(args.bg,f'Cam_{str(i).zfill(2)}_{pix}.png'))
                bg_alpha =  (1-(nmask)[...,np.newaxis])*bg
                png_alpha = png_alpha[...,:3] + bg_alpha[...,:3]
                imwrite(os.path.join(os.path.join(alpha_path,"bg"),f'{str(count).zfill(2)}.png'),png_alpha.astype('uint8'))
            
        else:
            imwrite(os.path.join(alpha_path,f'{str(i).zfill(5)}.exr'),relit.astype('float32'))
            png_relit = np.clip(np.power(np.clip(relit,0,None)+1e-5,0.45),0,1)*255
            imwrite(os.path.join(alpha_path,f'{str(i).zfill(5)}.png'),png_relit.astype('uint8'))
        del relit
        count += 1
    
    

