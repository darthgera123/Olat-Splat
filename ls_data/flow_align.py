import cv2
import sys
sys.path.insert(0, './RAFT/core')
import torch
from argparse import ArgumentParser
import argparse
from imageio.v2 import imread,imwrite
from utils.utils import InputPadder
from raft import RAFT
from utils import flow_viz
sys.path.remove('./RAFT/core')
import os
import numpy as np
from tqdm import tqdm

def parse_args():
    parser =  ArgumentParser(description="convert calib file to nerf format transforms.json")
    
    parser.add_argument("--input", default="", help="specify calib file location")
    parser.add_argument("--output", default="transforms.json", help="output path")
    parser.add_argument('--model',default="RAFT/models/raft-things.pth", help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    return args
# Get an optical flow model. As as example, we will use RAFT Small
# with the weights pretrained on the FlyingThings3D dataset
def load_image(path):
    # image = cv2.imread(path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = imread(path)
    h,w,c = image.shape
    image = cv2.resize(image,(int(w/4),int(h/4)),interpolation=cv2.INTER_AREA)
    return image
def pad_image(frame1,frame2):
    frame1 = torch.from_numpy(frame1).permute(2, 0, 1).float()
    frame2 = torch.from_numpy(frame2).permute(2, 0, 1).float()
    padder = InputPadder(frame1.shape)
    frame1, frame2 = padder.pad(frame1, frame2)
    frame1 = frame1.unsqueeze(0).cuda()
    frame2 = frame2.unsqueeze(0).cuda()
    return frame1, frame2

def align_images(image1, image2, flow_map):
    # Get the dimensions of the images
    h, w, _ = image1.shape
    
    

    # Calculate the grid of coordinates for the resized flow map
    flow_grid_x, flow_grid_y = np.meshgrid(np.arange(w), np.arange(h))
    flow_grid_x = flow_grid_x + flow_map[:, :, 0]
    flow_grid_y = flow_grid_y + flow_map[:, :, 1]

    # Remap (warp) image2 based on the resized flow map
    flow_grid = np.stack([flow_grid_x, flow_grid_y], axis=-1)
    warp_matrix = flow_grid.astype(np.float32)

    # Warp the image using the affine transformation
    warped_image = cv2.remap(image2, warp_matrix, None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return warped_image

if __name__ == '__main__':
    
    args = parse_args()
    full_lights = [1,15,35,56,77,98,119,140,161,182,224,245,266,287,308,329,350,363]
    canon = [203]
    bright = [363]
    os.makedirs(args.output,exist_ok=True)
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.cuda()
    model.eval()
    canon_track = os.path.join(args.input,f'L_{str(canon[0]).zfill(3)}')
    
    
    for light in tqdm(full_lights):
        print(light)
        input_track1 = os.path.join(args.input,f'L_{str(light).zfill(3)}')
        output_track = os.path.join(args.output,f'L_{str(light).zfill(3)}')
        flow_dir = os.path.join(args.output,f'flow/{str(canon[0])}_{str(light).zfill(3)}')
        os.makedirs(flow_dir,exist_ok=True)
        os.makedirs(output_track,exist_ok=True)
        for cam in tqdm(range(0,40)):
            frame1 = load_image(os.path.join(canon_track,f'Cam_{str(cam).zfill(2)}.exr'))
            frame2 = load_image(os.path.join(input_track1,f'Cam_{str(cam).zfill(2)}.exr'))
            image1 = frame1
            image2 = frame2
            if light in bright:
                frame2 = frame2*5
                image2 = image2 * 5 #match the brightness for effective flow compute
            frame1,frame2 = pad_image(frame1,frame2)
            flow_low, flow_up = model(frame1, frame2, iters=20, test_mode=True)
            flow = flow_up[0].permute(1, 2, 0).detach().cpu().numpy()
            # Resize the flow map to match the image size
    
            flow_map_resized = flow[:,3:-3]

            np.save(os.path.join(flow_dir,f'Cam_{str(cam).zfill(2)}.npy'),flow_map_resized)
            # cv2.imwrite(os.path.join(args.output,f'Flow_{str(i).zfill(2)}.jpg'),flo[:, :, [2,1,0]])
            aligned_image = align_images(image1,image2,flow_map_resized)
            del image1,image2,frame1,frame2
            # cv2.imwrite(os.path.join(args.output,f'Cam_{str(i).zfill(2)}.jpg'),cv2.cvtColor(aligned_image, cv2.COLOR_RGB2BGR))
            
            imwrite(os.path.join(output_track,f'Cam_{str(cam).zfill(2)}.exr'),aligned_image.astype('float32'))
    
    for light in tqdm(canon):
        input_track1 = os.path.join(args.input,f'L_{str(light).zfill(3)}')
        output_track = os.path.join(args.output,f'L_{str(light).zfill(3)}')
        os.makedirs(output_track,exist_ok=True)
        for cam in tqdm(range(0,40)):
            image = load_image(os.path.join(canon_track,f'Cam_{str(cam).zfill(2)}.exr'))
            imwrite(os.path.join(output_track,f'Cam_{str(cam).zfill(2)}.exr'),image.astype('float32'))

