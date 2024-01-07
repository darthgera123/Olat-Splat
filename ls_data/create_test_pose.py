from sys import argv
from argparse import ArgumentParser
import os
import numpy as np
from tqdm import tqdm
import json
from imageio.v2 import imread,imwrite
from scipy.spatial.transform import Rotation as R
import cv2

"""
python create_test_pose.py --input_json /CT/prithvi2/static00/3dgs/data/sandhoodie/full_light/transforms_train.json \
    --start_pose 14 --mid_pose 5 --end_pose 26 --output /CT/prithvi2/static00/3dgs/data/sandhoodie/full_light
"""

def parse_args():
    parser =  ArgumentParser(description="convert calib file to nerf format transforms.json")
    parser.add_argument("--input_json", default="transforms_train", help="input_json_file")
    parser.add_argument("--start_pose", default=1,type=int, help="input_json_file")
    parser.add_argument("--mid_pose", default=2,type=int, help="input_json_file")
    parser.add_argument("--end_pose", default=2,type=int, help="input_json_file")
    parser.add_argument("--num", default=5,type=int, help="number of poses")
    parser.add_argument("--output", default="transforms_train", help="input_json_file")
    parser.add_argument("--scale", default=1,type=int, help="scale")
    args = parser.parse_args()
    return args

def decompose_matrix(matrix):
    """
    Decomposes a 4x4 transformation matrix into translation and rotation components.
    """
    translation = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3]
    rotation = R.from_matrix(rotation_matrix).as_quat()
    return translation, rotation

def slerp(quat1, quat2, t):
    """
    Spherical linear interpolation (SLERP) between two quaternions.
    """
    dot = np.dot(quat1, quat2)

    # If the dot product is negative, slerp won't take the shorter path.
    # Fix by reversing one quaternion.
    if dot < 0.0:
        quat2 = -quat2
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # If the inputs are too close for comfort, linearly interpolate
        result = quat1 + t * (quat2 - quat1)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)  # angle between input vectors
    theta = theta_0 * t  # angle between v0 and result

    quat3 = quat2 - quat1 * dot
    quat3 = quat3 / np.linalg.norm(quat3)

    return quat1 * np.cos(theta) + quat3 * np.sin(theta)

def interpolate_pose(matrix1, matrix2, t):
    """
    Interpolates between two 4x4 transformation matrices.
    """
    trans1, rot1 = decompose_matrix(matrix1)
    trans2, rot2 = decompose_matrix(matrix2)

    interp_trans = (1 - t) * trans1 + t * trans2
    interp_rot = slerp(rot1, rot2, t)
    interp_rot_matrix = R.from_quat(interp_rot).as_matrix()

    interp_matrix = np.eye(4)
    interp_matrix[:3, :3] = interp_rot_matrix
    interp_matrix[:3, 3] = interp_trans

    return interp_matrix


def generate_interpolated_poses(matrix1, matrix2, N):
    """
    Generates N interpolated poses between two transformation matrices.
    """
    interpolated_poses = []
    for i in range(N + 1):
        t = i / N
        interpolated_pose = interpolate_pose(matrix1, matrix2, t)
        interpolated_poses.append(interpolated_pose)

    return interpolated_poses

def generate_circle_poses(center, radius, num_cameras):
    """
    Generate camera poses that create a circle along the X-axis.
    
    Args:
    - center: The center point of the circle (3D coordinates).
    - radius: The radius of the circle.
    - num_cameras: The number of camera poses to generate.
    - intrinsics: The camera intrinsic matrix (3x3).

    Returns:
    - A list of camera poses (extrinsic matrices).
    """

    circle_poses = []

    for i in range(num_cameras):
        angle = (2 * np.pi * i) / num_cameras  # Calculate angle for each camera
        x = center[0] + radius * np.cos(angle)  # X-coordinate on the circle
        y = center[1]  # Y-coordinate (no change along the Y-axis)
        z = center[2] + radius * np.sin(angle)  # Z-coordinate on the circle

        # Create the camera's extrinsic matrix
        translation = np.array([x, y, z])
        rotation = np.eye(3)  # No rotation for simplicity
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = rotation
        extrinsic_matrix[:3, 3] = translation

        # Apply camera intrinsics
        # camera_matrix = intrinsics @ extrinsic_matrix[:3, :]
        # extrinsic_matrix[:3, :] = camera_matrix

        circle_poses.append(extrinsic_matrix)

    return circle_poses

def load_json(json_path):
    with open(json_path, 'r') as h:
        data = json.load(h)
    return data  

def save_json(data,json_path):
    with open(json_path,'w') as file:
        json.dump(data, file, sort_keys=True, indent=4)
def mirror_camera_extrinsics(extrinsic_matrix):
    # Make a copy of the input matrix to avoid modifying it directly
    mirrored_extrinsic = extrinsic_matrix.copy()

    # Invert the sign of the translation along the X-axis
    mirrored_extrinsic[0, 3] = -mirrored_extrinsic[0, 3]

    # Flip the rotation along the Z-axis
    mirrored_extrinsic[:3, :3] = np.dot(np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]), mirrored_extrinsic[:3, :3])

    return mirrored_extrinsic

if __name__ == '__main__':
    
    args = parse_args()
    os.makedirs(args.output,exist_ok=True)
    transforms = load_json(args.input_json)

    transforms_test = dict()
    transforms_test["aabb_scale"] = 16.0
    
    N=args.num
    start_frame = transforms["frames"][args.start_pose]
    pose1 = np.asarray(start_frame["transform_matrix"])
    # pose2 = np.asarray(transforms["frames"][args.mid_pose]["transform_matrix"])
    # pose3 = np.asarray(transforms["frames"][args.end_pose]["transform_matrix"])
    # pose2 = mirror_camera_extrinsics(pose1)
    # pose4 = mirror_camera_extrinsics(pose3)
    # poses = generate_interpolated_poses(pose1,pose3,N)
    center = [0.64895969, 0.61341898, 1.64698363]
    poses =  generate_circle_poses(center,1,10)
            
    
    frames = []
    for i in range(0,len(poses)):
        frame={}
        frame = start_frame.copy()
        frame["transform_matrix"] = poses[i].tolist()
        frames.append(frame)
            
        
    
    transforms_test["frames"] = frames
    save_json(transforms_test,os.path.join(args.output,'transforms_test.json'))


    