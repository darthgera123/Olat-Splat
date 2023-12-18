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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel_exr
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.sh_utils import RGB2SH,SH2RGB
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    # lp = ModelParams(parser)
    # op = OptimizationParams(parser)
    # pp = PipelineParams(parser)
    
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument("--gauss_base", type=str, default = None)
    parser.add_argument("--gauss1_path", type=str, default = None)
    parser.add_argument("--gauss2_path", type=str, default = None)
    parser.add_argument("--save_dir", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)
    save = os.path.join(args.save_dir,'point_cloud/iteration_7000/')
    os.makedirs(save,exist_ok=True)
    # print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    gauss1 =  GaussianModel_exr(sh_degree=3)
    gauss2 = GaussianModel_exr(sh_degree=3)
    gauss1.load_ply(args.gauss1_path)
    # print(gauss1._features_dc.shape)
    gauss2.load_ply(args.gauss2_path)
    gauss1.detach_prop()
    gauss2.detach_prop()
    gauss3 = GaussianModel_exr(sh_degree=3)
    # gauss3 = gauss2
    gauss3.load_ply(args.gauss_base)
    gauss3.detach_prop()
    # g1 = gauss1.get_opacity
    # g2 = gauss2.get_opacity
    # g3 = (1-g1)*g2
    # print(gauss3._scaling.mean(1))
    
    cyan = torch.tensor([[0,1,1]]).cuda()
    yellow = torch.tensor([[1,1,0]]).cuda()
    black = torch.tensor([[0,0,0]]).cuda()
    off = torch.tensor([[0.5,0.5,0.5]]).cuda()
    white = torch.tensor([[1,1,1]]).cuda()
    rgb1 = torch.pow(SH2RGB(gauss1._features_dc),2.2)*cyan
    sh1 = RGB2SH(torch.pow(rgb1,0.45))
    rgb2 = torch.pow(SH2RGB(gauss2._features_dc),2.2)*yellow
    sh2 = RGB2SH(torch.pow(rgb2,0.45))
    rgb3 = rgb1+rgb2
    sh3 = RGB2SH(torch.pow(rgb3,0.45))
    gauss3._features_dc = sh3
    # gauss3._features_dc = gauss3._features_dc*white
    # print(torch.abs(gauss3._features_rest - gauss1._features_rest))
    # print(gauss3._features_rest.mean(axis=1))
    # print(gauss1._features_rest.mean(axis=1))
    # gauss3._features_rest = gauss1._features_rest*yellow + gauss2._features_rest*cyan
    # gauss3._features_rest = gauss1._features_rest*yellow + gauss2._features_rest*cyan
    # print(gauss3._features_rest.mean(axis=1))
    # gauss3._features_rest = gauss1._features_rest*cyan + gauss2._features_rest*yellow
    gauss3._features_rest = gauss3._features_rest*black
    # gauss3._opacity = g1 + g2
    # print(torch.max(gauss3._features_rest-gauss1._features_rest))
    gauss3.save_ply(os.path.join(save,'point_cloud.ply'))
    
