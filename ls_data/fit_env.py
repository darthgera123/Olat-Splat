from argparse import ArgumentParser

import torch
import torch.nn as nn
import numpy as np
from imageio.v2 import imread,imwrite
import cv2
import os
import json
from tqdm import tqdm
TINY_NUMBER = 1e-8


def parse_raw_sg(sg):
    SGLobes = sg[..., :3] / (torch.norm(sg[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)  # [..., M, 3]
    SGLambdas = torch.abs(sg[..., 3:4])
    SGMus = torch.abs(sg[..., -3:])
    return SGLobes, SGLambdas, SGMus



#######################################################################################################
# compute envmap from SG
#######################################################################################################
# def SG2Envmap(lgtSGs, H, W, upper_hemi=False):
#     # exactly same convetion as Mitsuba, check envmap_convention.png
#     if upper_hemi:
#         phi, theta = torch.meshgrid([torch.linspace(0., np.pi/2., H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])
#     else:
#         phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])

#     viewdirs = torch.stack([torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi)],
#                            dim=-1)    # [H, W, 3]
#     # print(viewdirs[0, 0, :], viewdirs[0, W//2, :], viewdirs[0, -1, :])
#     # print(viewdirs[H//2, 0, :], viewdirs[H//2, W//2, :], viewdirs[H//2, -1, :])
#     # print(viewdirs[-1, 0, :], viewdirs[-1, W//2, :], viewdirs[-1, -1, :])

#     # lgtSGs = lgtSGs.clone().detach()
#     viewdirs = viewdirs.to(lgtSGs.device)
#     viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]
#     # [M, 7] ---> [..., M, 7]
#     dots_sh = list(viewdirs.shape[:-2])
#     M = lgtSGs.shape[0]
#     lgtSGs = lgtSGs.view([1,]*len(dots_sh)+[M, 7]).expand(dots_sh+[M, 7])
#     # sanity
#     # [..., M, 3]
#     lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)
#     lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
#     lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values
#     # [..., M, 3]
#     rgb = lgtSGMus * torch.exp(lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
#     rgb = torch.sum(rgb, dim=-2)  # [..., 3]
#     envmap = rgb.reshape((H, W, 3))
#     return envmap

def SG2Envmap(lgtLobes,lgtLambda,lgtMu, H, W, upper_hemi=False):
    # exactly same convetion as Mitsuba, check envmap_convention.png
    if upper_hemi:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi/2., H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])
    else:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])

    viewdirs = torch.stack([torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi)],
                           dim=-1)    # [H, W, 3]
    # print(viewdirs[0, 0, :], viewdirs[0, W//2, :], viewdirs[0, -1, :])
    # print(viewdirs[H//2, 0, :], viewdirs[H//2, W//2, :], viewdirs[H//2, -1, :])
    # print(viewdirs[-1, 0, :], viewdirs[-1, W//2, :], viewdirs[-1, -1, :])

    # lgtSGs = lgtSGs.clone().detach()
    viewdirs = viewdirs.to(lgtLobes.device)
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]
    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])
    M = lgtLobes.shape[0]
    lgtLobes = lgtLobes.view([1,]*len(dots_sh)+[M, 3]).expand(dots_sh+[M, 3])
    # sanity
    # [..., M, 3]
    lgtSGLobes = lgtLobes[..., :3] / (torch.norm(lgtLobes[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)
    lgtSGLambdas = torch.abs(lgtLambda)
    lgtSGMus = torch.abs(lgtMu)  # positive values
    # [..., M, 3]
    rgb = lgtSGMus * torch.exp(lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    
    rgb = torch.sum(rgb, dim=-2)  # [..., 3]
    
    envmap = rgb.reshape((H, W, 3))
    return envmap


# def SG2Envmap(lgtSGs, H, W):
#     numLgtSGs = lgtSGs.shape[0]

#     phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), torch.linspace(0.0, 2 * np.pi, W)])
#     viewdirs = torch.stack((torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi)),
#                            dim=2).cuda()

#     viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]

#     # [n_envsg, 7]
#     sum_sg2 = torch.cat(parse_raw_sg(lgtSGs), dim=-1)

#     # [..., n_envsg, 7]
#     sh = list(viewdirs.shape[:-2])
#     sum_sg2 = sum_sg2.view([1, ] * len(sh) + [numLgtSGs, 7]).expand(sh + [-1, -1])

#     # [..., n_envsg, 3]
#     rgb = sum_sg2[..., -3:] * torch.exp(sum_sg2[..., 3:4] *
#                                         (torch.sum(viewdirs * sum_sg2[..., :3], dim=-1, keepdim=True) - 1.))
#     rgb = torch.sum(rgb, dim=-2)  # [..., 3]

#     env_map = rgb.reshape((H, W, 3))

#     return env_map
def load_json(json_path):
    with open(json_path, 'r') as h:
        data = json.load(h)
    return data

def save_json(data,json_path):
    with open(json_path,'w') as file:
        json.dump(data, file, sort_keys=True, indent=4)


def parse_args():
    parser =  ArgumentParser(description="convert calib file to nerf format transforms.json")
    parser.add_argument("--position", default="", help="specify calib file location")
    parser.add_argument("--env", default="", help="specify calib file location")
    parser.add_argument("--output", default="transforms.json", help="output path")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    os.makedirs(args.output,exist_ok=True)

    # load ground-truth envmap
    filename = args.env
    filename = os.path.abspath(filename)
    gt_envmap = imread(filename)[:,:,:3]
    gt_envmap = cv2.resize(gt_envmap, (512, 256), interpolation=cv2.INTER_AREA)
    gt_envmap = torch.from_numpy(gt_envmap).cuda()
    H, W = gt_envmap.shape[:2]


    lpos = np.asarray(load_json(args.position)['light_dir'])[:,[1,2,0]]
    H,W = 256,512
    numLgtSGs = lpos.shape[0]
    lgtLobes = torch.from_numpy(lpos).cuda()  # lobe + lambda + mu
    lgtLambda = nn.Parameter(torch.randn(numLgtSGs, 1).cuda()).requires_grad_(True)  # lobe + lambda + mu
    lgtMu = nn.Parameter(torch.randn(numLgtSGs, 3).cuda()).requires_grad_(True)  # lobe + lambda + mu

    

    optimizer = torch.optim.Adam([lgtLambda,lgtMu], lr=1e-2)

    N_iter = 100000

    pbar = tqdm(total=N_iter, desc="Training", unit="step")

    for step in range(N_iter):
        optimizer.zero_grad()
        env_map = SG2Envmap(lgtLobes,lgtLambda,lgtMu, H, W)
        loss = torch.mean((env_map - gt_envmap) * (env_map - gt_envmap))
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            envmap_check = env_map.clone().detach().cpu().numpy()
            gt_envmap_check = gt_envmap.clone().detach().cpu().numpy()
            im = np.concatenate((gt_envmap_check, envmap_check), axis=0)
            im = np.power(im, 1./2.2)
            im = np.clip(im, 0., 1.)
            # im = (im - im.min()) / (im.max() - im.min() + TINY_NUMBER)
            im = np.uint8(im * 255.)
            # print('step: {}, loss: {}'.format(step, loss.item()))
            # print(f"Step {step}, Loss: {loss.item():.2f}")
            pbar.set_description(f"Training (Loss: {loss.item():.2f})")
            pbar.update(100)
            imwrite(os.path.join(args.output, 'sgfit_cath_{}.png'.format(numLgtSGs)), im.astype('uint8'))
            imwrite(os.path.join(args.output, 'sgfit_cath_{}.hdr'.format(numLgtSGs)), envmap_check.astype('float32'))
            lgtLambda_save = lgtLambda.clone().detach().cpu().numpy()
            lgtMu_save = lgtLambda.clone().detach().cpu().numpy()
            lgtLobes_save = lgtLobes.clone().cpu().numpy()
            lgtSG = np.concatenate([lgtLobes_save,lgtLambda_save,lgtMu_save],axis=1)
            np.save(os.path.join(args.output, 'sg_cath_{}.npy'.format(numLgtSGs)), lgtSG)