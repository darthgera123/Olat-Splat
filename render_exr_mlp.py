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

import torch
from scene import Scene, SpecularModel, Scene_exr_light
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render,render_spec
import torchvision
from utils.general_utils import safe_state
# from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_model import GaussianModel,GaussianModel_exr
from imageio.v2 import imread,imwrite
import numpy as np
import time
def torch2numpy(img):
    np_img = img.cpu().detach().numpy()
    np_img = np.transpose(np_img, (1, 2, 0))
    
    return np_img

def exr2png(img):
    np_img = np.clip(np.power(np.clip(img,0,None),0.45),0,1)*255
    return np_img.astype('uint8')

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, specular):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_color")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    # depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    # acc_path = os.path.join(model_path, name, "ours_{}".format(iteration), "acc")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    # makedirs(depth_path, exist_ok=True)
    # makedirs(acc_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        dir_pp = (gaussians.get_xyz - view.camera_center.repeat(gaussians.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        light_dir_pp = (gaussians.get_xyz - view.light_dir.repeat(gaussians.get_features.shape[0], 1))
        light_dir_pp_normalized = light_dir_pp / dir_pp.norm(dim=1, keepdim=True)
        mlp_color = specular.step(gaussians.get_xyz.detach(), dir_pp_normalized,light_dir_pp_normalized)
        start_time = time.time()
        results = render_spec(view, gaussians, pipeline, background, mlp_color)
        rendering = results["render"]
        end_time = time.time()

# Calculate the difference in seconds
        execution_time = end_time - start_time

        # print(f"The function took {execution_time} seconds to execute.")
        # depth = results["depth"]
        # depth = depth / (depth.max() + 1e-5)

        gt = view.original_image[0:3, :, :]
        # np_img = np.power(np_img,2.2)
        # np_render = np.power(torch2numpy(rendering),2.2)
        np_render = torch2numpy(rendering)
        np_gt = torch2numpy(gt)
        imwrite(os.path.join(render_path, '{0:05d}'.format(idx) + ".exr"),np_render)
        imwrite(os.path.join(gts_path, '{0:05d}'.format(idx) + ".exr"),np_gt)
        
        imwrite(os.path.join(render_path, '{0:05d}'.format(idx) + ".png"),exr2png(np_render))
        imwrite(os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"),exr2png(np_gt))
        
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        del gt,rendering,np_render,np_gt


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                mode: str):
    with torch.no_grad():
        gaussians = GaussianModel_exr(dataset.sh_degree)
        scene = Scene_exr_light(dataset, gaussians, load_iteration=iteration, shuffle=False)
        specular = SpecularModel(model='hash')
        specular.load_weights(dataset.model_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_func = render_set

        if not skip_train:
            render_func(dataset.model_path, "train", scene.loaded_iter,
                        scene.getTrainCameras(), gaussians, pipeline,
                        background, specular)

        if not skip_test:
            render_func(dataset.model_path, "test", scene.loaded_iter,
                        scene.getTestCameras(), gaussians, pipeline,
                        background, specular)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'view', 'all', 'pose', 'original'])
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mode)
