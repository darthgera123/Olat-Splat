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
    parser.add_argument("--points", default="", help="specify calib file location")
    parser.add_argument("--colors", default="transforms.json", help="output path")
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

def generate_env_map(env_map,points, width, height, center, colors):
    env_map = np.zeros((height, width, 3), dtype=np.uint8)
    coords = []
    center_x, center_y, center_z = center
    for i, point in enumerate(points):
        # env_map = np.zeros((height, width, 3), dtype=np.uint8)
        x, y, z = point[0] - center_x, point[1] - center_y, point[2] - center_z
        r, theta, phi = cartesian_to_spherical(x, y, z)
        
        u, v = spherical_to_equirectangular(theta, phi, width, height)
        coords.append((u,v))
        # print(colors[i]/16)
        # cv2.circle(env_map, (u, v), 5, (colors[i]/16), -1)
        cv2.circle(env_map, (u, v), 4, colors[i], -1)
        # imwrite(f'envmap/envmap_{str(i).zfill(3)}.png',np.flipud(env_map).astype('uint8'))

    return env_map,coords

def max_int(image, vor, centers):
    # Create a KDTree for efficient nearest neighbor search
    tree = cKDTree(vor.vertices)

    # Compute the sum of RGB values for each pixel (assumes image shape is height x width x 3)
    intensity_sum = np.sum(image, axis=2)

    # Pre-compute the grid of pixel coordinates
    xx, yy = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    pixel_coords = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # Find the indices of the nearest Voronoi cell for each pixel
    _, nearest_cell_indices = tree.query(pixel_coords)

    # Initialize arrays to store the maximum intensity and corresponding pixel for each cell
    max_intensities = np.zeros(len(centers))
    max_intensity_pixels = np.zeros((len(centers), 3))  # Assuming a 3-channel (RGB) image

    # Flatten the image array and intensity sum for vectorized operations
    flat_image = image.reshape(-1, 3)
    flat_intensity_sum = intensity_sum.ravel()

    # Iterate over each cell
    for i in range(len(centers)):
        # Find pixels belonging to this cell and their intensities
        cell_pixel_indices = np.where(nearest_cell_indices == i)[0]
        cell_pixel_intensities = flat_intensity_sum[cell_pixel_indices]

        # Find the pixel with the maximum intensity
        if cell_pixel_intensities.size > 0:
            max_intensity_index = cell_pixel_indices[np.argmax(cell_pixel_intensities)]
            max_intensities[i] = flat_intensity_sum[max_intensity_index]
            max_intensity_pixels[i] = flat_image[max_intensity_index]

    return max_intensity_pixels



if __name__ == '__main__':
    
    args = parse_args()
    os.makedirs(args.output,exist_ok=True)
    # points = np.loadtxt(args.points)
    points = np.asarray(load_json(args.points)['light_dir'])
    # colors = np.loadtxt(args.colors)
    colors = [(255,255,255) for _ in range(331)]
    # print(colors)
    
    center = points.mean(axis = 0)
    print(center)
    # points = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0)]

    width, height = 1024, 512
    if args.envmap == "":
        envmap = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        # scaling to be figure out
        envmap_og = imread(args.envmap)
        
        envmap = (np.clip(np.power(envmap_og,0.45),0,1)*255)
    env_map,coords = generate_env_map(envmap,points, width, height, center, colors)
    vor = Voronoi(coords)
    # for simplex in vor.ridge_vertices:
    #     simplex = np.asarray(simplex)
    #     if np.all(simplex >= 0):
    #         start_point = tuple([int(v) for v in vor.vertices[simplex[0]]])  # Convert to integer
    #         end_point = tuple([int(v) for v in vor.vertices[simplex[1]]]) 
    #         # print(start_point,end_point)
    #         cv2.line(env_map, start_point, end_point, (0, 0, 255), 2)
    # cv2.imwrite('./envma2_neg.png', np.flipud(env_map))
    imwrite('envmap.hdr',env_map.astype('float32'))
    # imwrite('env_image_with_voronoi.jpg', env_map.astype('uint8'))

    weighted_average = max_int(envmap,vor,coords)
    colors = [(int(w[0]),int(w[1]),int(w[2])) for w in weighted_average]
    # env_map,coords = generate_env_map(np.zeros((height, width, 3), dtype=np.uint8),points, width, height, center,colors)
    # imwrite('envmap_vor.hdr',env_map.astype('float32'))