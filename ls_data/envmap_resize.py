import numpy as np
import cv2
from argparse import ArgumentParser
import os
import numpy as np
import json
from imageio.v2 import imread,imwrite
from tqdm import tqdm

def load_json(json_path):
    with open(json_path, 'r') as h:
        data = json.load(h)
    return data

def save_json(data,json_path):
    with open(json_path,'w') as file:
        json.dump(data, file, sort_keys=True, indent=4)

def parse_args():
    parser =  ArgumentParser(description="convert calib file to nerf format transforms.json")
    parser.add_argument("--map", default="", help="specify calib file location")
    parser.add_argument("--order", default="transforms.json", help="output path")
    parser.add_argument("--points", default="transforms.json", help="output path")
    parser.add_argument("--output", default="transforms.json", help="output path")
    parser.add_argument("--input", default="", help="specify calib file location")
    parser.add_argument("--median", default="transforms.json", help="output path")
    args = parser.parse_args()
    return args

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x) # Azimuth
    phi = np.arccos(z / r)   # Elevation
    return r, theta, phi

def spherical_to_equirectangular(theta, phi, width, height):
    u = width - (0.5 * (theta / np.pi + 1) * width)
    v = height * phi / np.pi
    return int(u), int(v)

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

def rotate_envmap(envmap,pixels=10):
    envmap_rolled = np.roll(envmap, shift=pixels, axis=1)
    return envmap_rolled


if __name__ == '__main__':
    
    args = parse_args()
    
    
    os.makedirs(args.output,exist_ok=True)
    os.makedirs(args.output+'/exr/',exist_ok=True)
    os.makedirs(args.output+'/png/',exist_ok=True)
    files = os.listdir(args.input)
    for file in tqdm(files):
        envmap = imread(os.path.join(args.input,file))
        envmap_name = (file).split('.')[0]
        envmap_resize = cv2.resize(envmap,(512,256),interpolation=cv2.INTER_AREA)
        envmap_median = np.median(imread(args.median))

        # envmap_resize = envmap_resize * 10
        # med = np.median(envmap_resize)
        envmap_resize_med = np.median(envmap_resize)
        print(envmap_median,envmap_resize_med)
        envmap_resize = (envmap_resize/envmap_resize_med)*envmap_median 
        
        imwrite(os.path.join(args.output+'/exr/',f'{envmap_name}.exr'),envmap_resize.astype('float32'))
        imwrite(f'{args.output}/png/{envmap_name}.png',(np.clip(np.power(envmap_resize,0.45),0,1)*255).astype('uint8'))
    # for pix in range(0,512,8):
    #     envmap_roll = rotate_envmap(envmap,pix)
    #     
    #     imwrite(os.path.join(args.output,f'{envmap_name}_{pix}.png'),(np.clip(np.power(envmap_roll,0.45),0,1)*255).astype('uint8'))
    
    
    