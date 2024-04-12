import cv2
import sys

from argparse import ArgumentParser
import argparse
from imageio.v2 import imread,imwrite
import os
import numpy as np
from tqdm import tqdm
from folder_file_util import FileLock,makeNewDir,get_sorted_basenames,clear_folder_content




def parse_args():
    parser =  ArgumentParser(description="convert calib file to nerf format transforms.json")
    # parser.add_argument("--track1", default="", help="specify calib file location")
    # parser.add_argument("--track2", default="transforms.json", help="output path")
    parser.add_argument("--input", default="", help="specify calib file location")
    parser.add_argument("--flow", default="transforms.json", help="output path")
    parser.add_argument("--output", default="transforms.json", help="output path")
    parser.add_argument('--model',default="RAFT/models/raft-things.pth", help="restore checkpoint")
    args = parser.parse_args()
    return args

def load_image(path):
    image = imread(path)
    h,w,c = image.shape
    image = cv2.resize(image,(int(w/4),int(h/4)),interpolation=cv2.INTER_AREA)
    return image

if __name__ == '__main__':
    
    args = parse_args()
    full_lights = [1,15,35,56,77,98,119,140,161,182,203,224,245,266,287,308,329,350,363]
    lights = [x for x in range(15, 364) if x not in full_lights]
    canon = [203]
    bright = [363]
    os.makedirs(args.output,exist_ok=True)

    canon_track = os.path.join(args.input,f'L_{str(canon[0]).zfill(3)}')
    
    makeNewDir('.cache')

    

    for light in tqdm(range(1,364)):
        input_track1 = os.path.join(args.input,f'L_{str(light).zfill(3)}')
        output_track = os.path.join(args.output,f'L_{str(light).zfill(3)}')
        # clear_folder_content(output_track)
        os.makedirs(output_track,exist_ok=True)
        
        for cam in tqdm(range(0,40)):
            lock_file = '.cache/resize_%s_%s.lock'%(light,cam)
            with FileLock(lock_file) as lock:
                if lock.has_lock:
                    num_of_files = len(get_sorted_basenames(output_track, ['*.exr']))
                    if num_of_files == 40:
                        print('%s was done, skipping'%lock_file)
                    else:
                        image = load_image(os.path.join(input_track1,f'Cam_{str(cam).zfill(2)}.exr'))
                        imwrite(os.path.join(output_track,f'Cam_{str(cam).zfill(2)}.exr'),image.astype('float32'))
    

