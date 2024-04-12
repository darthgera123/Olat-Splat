import numpy as np
import cv2
from argparse import ArgumentParser
import os
import numpy as np
import json
from imageio.v2 import imread,imwrite
import torch
from skimage.metrics import structural_similarity as compute_ssim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from lpips import LPIPS

def load_json(json_path):
    with open(json_path, 'r') as h:
        data = json.load(h)
    return data

def save_json(data,json_path):
    with open(json_path,'w') as file:
        json.dump(data, file, sort_keys=True, indent=4)

def parse_args():
    parser =  ArgumentParser(description="convert calib file to nerf format transforms.json")
    parser.add_argument("--input", default="", help="specify calib file location")
    parser.add_argument("--comp", default="", help="specify calib file location")
    parser.add_argument("--order", default="transforms.json", help="output path")
    parser.add_argument("--points", default="transforms.json", help="output path")
    parser.add_argument("--output", default="transforms.json", help="output path")
    parser.add_argument("--envmap", default="", help="specify calib file location")
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    gt_folder = os.path.join(args.input,'gt')
    pred_folder = os.path.join(args.input,args.comp)

    envmaps = ['bedroom', 'cathedral', 'corridor', 'uffizi']

    cams = ['Cam_05','Cam_18','Cam_38']
    

    # Create an instance of the LPIPS class
    compute_lpips = LPIPS()

    for envmap in envmaps:
        psnr = 0
        ssim = 0
        lpips = 0
        for cam in cams:
            gt = imread(os.path.join(gt_folder, f'{cam}_{envmap}.png'))
            pred = imread(os.path.join(pred_folder, f'{cam}_{envmap}.png'))  # Assuming you have a pred_folder
           
            psnr += compute_psnr(gt, pred)
            ssim += compute_ssim(gt, pred,win_size=3)
            lpips += compute_lpips(torch.tensor(gt).permute(2,0,1), torch.tensor(pred).permute(2,0,1)).item()
        num_cams = len(cams)
        print(f'Envmap {envmap}, PSNR: {psnr/num_cams:.2f}, SSIM: {ssim/num_cams:.4f}, LPIPS: {lpips/num_cams:.4f}')
