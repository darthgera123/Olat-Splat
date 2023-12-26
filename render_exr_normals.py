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
from scene import Scene, Scene_exr
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel,GaussianModel_exr
from imageio.v2 import imread,imwrite
import numpy as np
import time
def torch2numpy_pow(img):
    np_img = img.cpu().detach().numpy()
    np_img = np.transpose(np_img, (1, 2, 0))
    np_img = np.power(np_img,2.2)
    return np_img

def torch2numpy(img):
    np_img = img.cpu().detach().numpy()
    np_img = np.transpose(np_img, (1, 2, 0))
    np_img = np.power(np_img,2.2)
    return np_img

def exr2png(img):
    np_img = np.clip(np.power(np.clip(img,0,None),0.45),0,1)*255
    return np_img.astype('uint8')

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    alpha_path = os.path.join(model_path, name, "ours_{}".format(iteration), "alpha")
    normal_ref_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal_ref")
    normal_axis_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal_axis")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(normal_ref_path, exist_ok=True)
    makedirs(normal_axis_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)
    makedirs(alpha_path, exist_ok=True)
    cyan = torch.tensor([[0,1,1]]).cuda().reshape((3,1,1))
    yellow = torch.tensor([[1,1,0]]).cuda().reshape((3,1,1))
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")): 
        start_time = time.time()
        results = render(view, gaussians, pipeline, background)
        end_time = time.time()
        execution_time = end_time - start_time

        print(f"The function took {execution_time} seconds to execute.")
        gt = view.original_image[0:3, :, :]
        
        np_gt = torch2numpy_pow(gt)
        rendering = results["render"]
        np_render = torch2numpy_pow(rendering)
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)
        alpha = results["alpha"]
        normal_ref = results["normal_ref"]
        normal_ref = 0.5 + (0.5*normal_ref)

        normal_axis = results["normal_axis"]
        normal_axis = 0.5 + (0.5*normal_axis)

        if "normal" in results.keys():
            normal = results["normal"]
            normal = 0.5 + (0.5*normal)
            torchvision.utils.save_image(normal, os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"))
        
        imwrite(os.path.join(render_path, '{0:05d}'.format(idx) + ".exr"),np_render)
        imwrite(os.path.join(gts_path, '{0:05d}'.format(idx) + ".exr"),np_gt)
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(alpha, os.path.join(alpha_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(normal_ref, os.path.join(normal_ref_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(normal_axis, os.path.join(normal_axis_path, '{0:05d}'.format(idx) + ".png"))
        
        imwrite(os.path.join(render_path, '{0:05d}'.format(idx) + ".png"),exr2png(np_render))
        imwrite(os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"),exr2png(np_gt))
        # imwrite(os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"),depth)
        # imwrite(os.path.join(alpha_path, '{0:05d}'.format(idx) + ".png"),alpha)
        # imwrite(os.path.join(normal_ref_path, '{0:05d}'.format(idx) + ".png"),normal_ref)
        # imwrite(os.path.join(normal_axis_path, '{0:05d}'.format(idx) + ".png"),normal_axis)
        # imwrite(os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"),normal)


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel_exr(dataset.sh_degree)
        if pipeline.brdf:
            gaussians.brdf = True
        scene = Scene_exr(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)