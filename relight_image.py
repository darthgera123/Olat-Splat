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

import os
import numpy as np
from argparse import ArgumentParser
import sys
from imageio.v2 import imread,imwrite
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument("--gauss_base", type=str, default = None)
    parser.add_argument("--image1_path", type=str, default = None)
    parser.add_argument("--image2_path", type=str, default = None)
    parser.add_argument("--save_dir", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)
    save = os.path.join(args.save_dir,'images')
    os.makedirs(save,exist_ok=True)
    cyan = np.array([0,1,1])
    yellow = np.array([1,1,0])
    for i in tqdm(range(0,40)):
        img1 = imread(os.path.join(args.image1_path,f'{str(i).zfill(5)}.exr'))
        img2 = imread(os.path.join(args.image2_path,f'{str(i).zfill(5)}.exr'))
        relit = img1*cyan + img2*yellow
        imwrite(os.path.join(save,f'Cam_{str(i).zfill(2)}.exr'),relit.astype('float32'))
        png_relit = np.clip(np.power(np.clip(relit,0,None)+1e-5,0.45),0,1)*255 
        imwrite(os.path.join(save,f'Cam_{str(i).zfill(2)}.png'),png_relit.astype('uint8'))