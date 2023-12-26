import numpy as np
import cv2
from imageio.v2 import imread,imwrite
import pyshtools
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def read_image(filename):
    """Read the image file and return its contents as a numpy array"""
    image = imread(filename)
    h,w,c = image.shape
    nw,nh = int(w/4),int(h/4)
    image = cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA)
    return imread(filename)

def latlong_to_spherical(img):
    """Convert a latitude-longitude image to spherical coordinates"""
    height, width, _ = img.shape
    c = np.zeros((2, height, width))
    for y in range(height):
        theta = np.pi * (y + 0.5) / height
        for x in range(width):
            phi = 2 * np.pi * x / width
            c[:, y, x] = img[y, x, :2] * np.sin(theta)
    return c

def project_to_sh(img, n):
    """Project each color channel of the image to SH basis of order n using pyshtools"""
    spherical_img = latlong_to_spherical(img)
    coeffs = []
    for i in range(3):  # Iterate over RGB channels
        channel_coeffs = pyshtools.expand.SHExpandDH(spherical_img[:, :, i], sampling=2,norm=4)
        coeffs.append(channel_coeffs)
    return np.asarray(coeffs)

def save_sh_projection(coeffs, width, height, filename):
    """Save the environment map reconstructed from SH coefficients"""
    image_reconstructed = np.zeros((height, width, 3))
    for y in range(height):
        theta = np.pi * (y + 0.5) / height
        for x in range(width):
            phi = 2 * np.pi * x / width
            for c in range(3):
                for l in range(coeffs.shape[1]):
                    for m in range(-l, l + 1):
                        image_reconstructed[y, x, c] += coeffs[c, l, m + l] * pyshtools.expand.MakeGridPoint(l, m, theta, phi)
    image_reconstructed = np.clip(image_reconstructed, 0, 255)
    imwrite(filename, image_reconstructed.astype(np.uint8))

def parse_args():
    parser =  ArgumentParser(description="project envmap to SH")
    parser.add_argument("--input_light", default="", help="specify calib file location")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()



    # Read the environment map
    filename = args.input_light  # Replace with your file path
    env_map = read_image(filename)

    # Project to SH basis of order n
    n = 8  # Example order
    sh_coeffs = project_to_sh(env_map, n)

    # Save the SH projected environment map
    output_filename = 'sh_projected_env_map.png'  # Output file name
    save_sh_projection(sh_coeffs, env_map.shape[1], env_map.shape[0], output_filename)