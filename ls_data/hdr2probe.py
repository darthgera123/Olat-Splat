import numpy as np
import cv2
from argparse import ArgumentParser
import os
import numpy as np
import json
from imageio.v2 import imread,imwrite
from scipy.spatial import Voronoi, cKDTree


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
    parser.add_argument("--envmap", default="", help="specify calib file location")
    
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

# def generate_env_map(env_map,points, width, height, center, colors):
#     # env_map = np.zeros((height, width, 3), dtype=np.uint8)
#     coords = []
#     center_x, center_y, center_z = center
#     for i, point in enumerate(points):
#         # env_map = np.zeros((height, width, 3), dtype=np.uint8)
#         x, y, z = point[0] - center_x, point[1] - center_y, point[2] - center_z
#         r, theta, phi = cartesian_to_spherical(x, y, z)
        
#         u, v = spherical_to_equirectangular(theta, phi, width, height)
#         coords.append((u,v))
#         # print(colors[i]/16)
#         # cv2.circle(env_map, (u, v), 5, (colors[i]/16), -1)
#         cv2.circle(env_map, (u, v), 3, colors[i], -1)
#         # imwrite(f'envmap/envmap_{str(i).zfill(3)}.png',np.flipud(env_map).astype('uint8'))
#     return env_map,coords

def generate_env_map(env_map, points, width, height, center, colors):
    center_x, center_y, center_z = center
    for point, color in zip(points, colors):
        x, y, z = point[0] - center_x, point[1] - center_y, point[2] - center_z
        r, theta, phi = cartesian_to_spherical(x, y, z)

        u, v = spherical_to_equirectangular(theta, phi, width, height)

        # Adjust the size and shape of the ellipsoid based on the phi angle
        max_radius = 5  # Maximum radius
        phi_normalized = phi / np.pi  # Normalize phi to [0, 1]

        # Calculate ellipsoid dimensions
        ellipsoid_width = max_radius
        ellipsoid_height = max_radius * (1 - abs(phi_normalized - 0.5) * 2)

        # Draw the ellipsoid
        cv2.ellipse(env_map, (u, v), (int(ellipsoid_width), int(ellipsoid_height)), 0, 0, 360, color, -1)
    
    return env_map


if __name__ == '__main__':
    
    args = parse_args()
    mapy = np.asarray(read_numbers_from_file(args.map)).reshape(256,512)
    order = np.asarray(read_numbers_from_file(args.order))
    points = np.asarray(load_json(args.points)['light_dir'])
    center = points.mean(axis = 0)
    # coords = generate_env_map(envmap,points, width, height, center, colors)

    envmap = imread(args.envmap)
    colors = []
    for i in range(331):
        mask = np.zeros((256,512,3))
        mask[mapy==(order[i]-1)] = 1
        # color = np.sum(envmap * mask)/(np.sum(mask==1))
        masked = envmap*mask
        intensity_sum = np.sum(masked, axis=2)
        max_intensity_location = np.unravel_index(np.argmax(intensity_sum), intensity_sum.shape)
        color = masked[max_intensity_location]
        
        colors.append(color)
    
    # colors = [(255,255,255) for _ in range(331)]
    overlay = np.zeros((256,512,3))
    envmap_ = generate_env_map(overlay,points,512,256,center,colors)

    imwrite('envmap.hdr',envmap_.astype('float32'))
    imwrite('envmap.png',(np.clip(np.power(envmap_,0.45),0,1)*255).astype('uint8'))
    imwrite('input.png',(np.clip(np.power(envmap,0.45),0,1)*255).astype('uint8'))

    