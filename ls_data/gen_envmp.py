import numpy as np
import cv2

import numpy as np
import cv2

def cartesian_to_spherical(x, z, y):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x) # Azimuth
    phi = np.arccos(z / r)   # Elevation
    return r, theta, phi

def spherical_to_equirectangular(theta, phi, width, height):
    u = width - (0.5 * (theta / np.pi + 1) * width)
    v = height * phi / np.pi
    return int(u), int(v)

def generate_env_map(points, width, height, center, colors):
    env_map = np.zeros((height, width, 3), dtype=np.uint8)

    center_x, center_y, center_z = center
    for i, point in enumerate(points):
        x, y, z = point[0] - center_x, point[1] - center_y, point[2] - center_z
        r, theta, phi = cartesian_to_spherical(x, y, z)

        u, v = spherical_to_equirectangular(theta, phi, width, height)

        print(colors[i]/16)
        cv2.circle(env_map, (u, v), 5, (colors[i]/16), -1)
        # cv2.circle(env_map, (u, v), 5, (255,255,255), -1)

    return env_map


points = np.loadtxt('./LSX_light_positions_aligned_new.xyz')
colors = np.loadtxt('./rgb_mapping_neg0.txt')
# print(colors)
print(len(points))
center = points.mean(axis = 0)
# points = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0)]

width, height = 1024, 512

env_map = generate_env_map(points, width, height, center, colors)

cv2.imwrite('./envma2_neg.png', env_map)
# cv2.waitKey(0)
# cv2.destroyAllWindows()