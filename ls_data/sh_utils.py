# -*- coding: utf-8 -*-
# @File    : utils.py


import math
import torch
import matplotlib
import numpy as np
import trimesh.triangles

from matplotlib import cm
from matplotlib import figure
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.axes_grid1 import make_axes_locatable

tiny_number = 1e-7


def plot_3d(samples):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")

    ax.scatter3D(samples[:, 0], samples[:, 1], samples[:, 2], color="green")
    plt.title("simple 3D scatter plot")

    plt.show()


def plot_2d(samples):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()

    ax.scatter(samples[:, 0], samples[:, 1], color="green")
    plt.title("simple 2D scatter plot")

    plt.show()


def sample_sphere(N: int):
    """
    Return Samples on Sphere
    :param N: Number of samples required on unit sphere
    :return: returns directions
    """
    eta_1, eta_2 = torch.rand(size=(N,)), torch.rand(size=(N,))

    z = 1 - 2 * eta_1
    cos_eta1 = torch.sqrt(torch.maximum(torch.Tensor([0.]), eta_1 * (1 - eta_1)))
    x, y = 2 * cos_eta1 * torch.cos(2 * math.pi * eta_2), 2 * cos_eta1 * torch.sin(2 * math.pi * eta_2)
    directions = torch.stack((x, y, z), dim=1)
    directions /= torch.linalg.norm(directions, dim=1, keepdims=True)
    return directions


def sample_uniform_hemisphere(N: int) -> np.array:
    """
    Samples N directions in Uniform Unit Sphere
    :param N: Required Number of Directions
    :return: returns directions
    """
    eta_1, eta_2 = np.random.uniform(low=0, high=1, size=N), np.random.uniform(low=0, high=1, size=N)

    z = eta_1
    phi = 2 * np.pi * eta_2

    x = np.cos(phi) * np.sqrt(np.maximum(0, 1 - z ** 2))
    y = np.sin(phi) * np.sqrt(np.maximum(0, 1 - z ** 2))

    return np.stack((x, y, z), axis=1)


def uniform_sphere_pdf() -> float:
    """
    PDF of sphere, integral of pdf over a sphere should be 1 as its a uniform distribution.
    from which we obtain c = 1/2pi which is pdf

    https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations (13.6.1)
    :return: pdf (1 / 2 * pi)
    """
    return 1 / (2 * np.round(np.pi, 7))


def get_spherical_coords(vec):
    """
    Input list of 3D vectors and the output is the np array of theta and phi values
    :param vec: list of 3D vectors (size, 3)
    :return: list of theta, phi values (size, 2)
    """
    device = 'cuda'
    v = vec / torch.linalg.norm(vec, axis=1).reshape(-1, 1)

    theta = torch.acos(v[:, 2]).to(device)  # cos inverse z
    # theta = np.arccos(v[:, 2])
    # phi = np.arctan2(v[:, 1], v[:, 0])  # tan inverse y / x

    inv_sin_theta = 1 / torch.sin(theta).to(device)
    # phi = np.arccos(np.clip(v[:, 0] * inv_sin_theta, -1.0, 1.0))
    phi = torch.acos(torch.clip(v[:, 0] * inv_sin_theta, -1.0, 1.0))
    phi = torch.where(v[:, 1] * inv_sin_theta < 0.0, 2.0 * np.pi - phi, phi)

    return theta, phi


def concentric_sample_disk(n):
    r1, r2 = torch.rand(n, dtype=torch.float32) * 2.0 - 1.0, torch.rand(n, dtype=torch.float32) * 2.0 - 1.0

    zero_x_y = torch.where(r1 == 0, True, False)
    zero_x_y = torch.logical_and(zero_x_y, torch.where(r2 == 0, True, False))
    zero_x_y = torch.stack((zero_x_y, zero_x_y), dim=1)
    zeros = torch.zeros((n, 2))

    c1, c2 = 4 * r1, 4 * r2
    x = torch.where(torch.abs(r1) > torch.abs(r2), torch.cos(np.pi * r2 / c1), torch.cos(np.pi / 2 - np.pi * r1 / c2))
    y = torch.where(torch.abs(r1) > torch.abs(r2), torch.sin(np.pi * r2 / c1), torch.sin(np.pi / 2 - np.pi * r1 / c2))

    r = torch.where(torch.abs(r1) > torch.abs(r2), r1, r2)
    r = torch.stack((r, r), dim=1)

    points = r * torch.stack((x, y), dim=1)

    return torch.where(zero_x_y, zeros, points)


def sample_hemisphere_cosine(n):
    disk_point = concentric_sample_disk(n)
    xx = disk_point[:, 0] ** 2
    yy = disk_point[:, 1] ** 2
    z = torch.FloatTensor([1]) - xx - yy
    z = torch.sqrt(torch.where(z < 0., torch.FloatTensor([0.]), z))

    wi_d = torch.cat((disk_point, torch.unsqueeze(z, dim=-1)), dim=-1)
    wi_d = wi_d / (torch.linalg.norm(wi_d, dim=1, keepdims=True) + 1e-8)

    return wi_d.cpu().numpy()


def get_uvs(points, triangle_vertices, triangle_uvs):
    barycentric_ratios = trimesh.triangles.points_to_barycentric(triangle_vertices, points)
    new_uvs = barycentric_ratios[:, 0][:, None] * triangle_uvs[:, 0, :] \
              + barycentric_ratios[:, 1][:, None] * triangle_uvs[:, 1, :] \
              + barycentric_ratios[:, 2][:, None] * triangle_uvs[:, 2, :]
    return new_uvs


if __name__ == '__main__':
    s = sample_sphere(5000)

    theta, phi = get_spherical_coords(s.cuda())
    plt.scatter(np.arange(0, theta.shape[0]), theta.cpu().numpy())
    plt.show()
    # plot_3d(s)
