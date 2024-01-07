from argparse import ArgumentParser

import torch
import torch.nn as nn
import numpy as np
from imageio.v2 import imread,imwrite
import cv2
import os
import json

TINY_NUMBER = 1e-8


def parse_raw_sg(sg):
    SGLobes = sg[..., :3] / (torch.norm(sg[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)  # [..., M, 3]
    SGLambdas = torch.abs(sg[..., 3:4])
    SGMus = torch.abs(sg[..., -3:])
    return SGLobes, SGLambdas, SGMus



#######################################################################################################
# compute envmap from SG
#######################################################################################################
def SG2Envmap(lgtSGs, H, W,lamda,mu, upper_hemi=False):
    # exactly same convetion as Mitsuba, check envmap_convention.png
    if upper_hemi:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi/2., H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])
    else:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])

    viewdirs = torch.stack([torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi)],
                           dim=-1)    # [H, W, 3]
    
    viewdirs = viewdirs.to(lgtSGs.device)
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]
    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])
    M = lgtSGs.shape[0]
    
    lgtSGs = lgtSGs.view([1,]*len(dots_sh)+[M, 7]).expand(dots_sh+[M, 7])
    
    # sanity
    # [..., M, 3]
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)
    lgtSGLambdas = lamda
    lgtSGMus = mu
    # [..., M, 3]
    rgb = lgtSGMus * torch.exp(lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = torch.sum(rgb, dim=-2)  # [..., 3]
    envmap = rgb.reshape((H, W, 3))
    return envmap




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
    parser.add_argument("--positions", default="", help="specify calib file location")
    parser.add_argument("--output", default="transforms.json", help="output path")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    os.makedirs(args.output,exist_ok=True)
    
    lpos = np.asarray(load_json(args.positions)['light_dir'])[:,[1,2,0]][0]
    H,W = 256,512
    numLgtSGs = lpos.shape[0]
    lgtSGs = torch.randn(numLgtSGs, 7).cuda()  # lobe + lambda + mu

    # Load positions
    lgtSGs[...,:3] = torch.from_numpy(lpos).cuda()
    
    env_map = SG2Envmap(lgtSGs, H, W,lamda=236.9705*2,mu=torch.ones(numLgtSGs,3).cuda(),upper_hemi=False)
    envmap_check = env_map.clone().detach().cpu().numpy()
    im = np.power(envmap_check, 1./2.2)
    im = np.clip(im, 0., 1.)*255
    imwrite(os.path.join(args.output, 'olat_{}.png'.format(numLgtSGs)), im.astype('uint8'))
    imwrite(os.path.join(args.output, 'olat_{}.hdr'.format(numLgtSGs)), envmap_check.astype('float32'))
    