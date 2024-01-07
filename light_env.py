# -*- coding: utf-8 -*-
# @Author  : dhawal1939
# @File    : light_env.py


import imageio
import torch
import numpy as np
from pathlib import Path
from config import device
from sh.legendre_sh import compute_sh


class EnvMap:
    def __init__(self, img_path: str = '', sh_order: int = 5, _format: str = 'PNG', img=None):
        """
        Holds data about the env_map and sh
        :param img_path: Img path
        :param sh_order: Order of SH coefficients required default 3
        :param _format: JPG PNG or HDR-FI
        """
        if img is None:
            self.image_path = Path(img_path)
            self.env_map = imageio.read(img_path, format=_format)  # shape is h, w, channel(RGB)
            self.env_map = np.array(self.env_map.get_data(0)).astype('float32') / 255.
            self.env_map = np.power(self.env_map, 2.2)
        else:
            self.env_map = img

        self.h, self.w, self.c = self.env_map.shape
        self.theta = (np.arange(self.h) + 0.5) * np.pi / self.h  # as the upper hemisphere only 0-180 degrees only
        self.phi = (np.arange(self.w) + 0.5) * 2 * np.pi / self.w  # around the point phi so 2 * 180 = 360 degrees

        dirs = np.transpose(np.meshgrid(self.theta, self.phi)).reshape(-1, 2)

        self.sh_base = compute_sh(torch.Tensor(dirs[:, 0]).to(device), torch.Tensor(dirs[:, 1]).to(device), sh_order)
        self.sh_base = self.sh_base.detach().cpu().numpy()
        theta = np.arange(self.h + 1) * np.pi / self.h
        '''
                d(\omega) == sin(\theta) d(\theta) d(\phi)
                d cos(\theta) / d(\theta) = - sin(\theta)
                sin(\theta) * d(\theta) = - d(cos(\theta))
            --> d(\omega) = - d(cos(\theta)) d(\phi)
        '''
        cos_theta = np.cos(theta)
        neg_d_cos = - (cos_theta[1:] - cos_theta[:-1])
        phi = np.arange(self.w + 1) * 2 * np.pi / self.w
        d_phi = phi[1:] - phi[0:-1]

        x, y = np.meshgrid(d_phi, neg_d_cos)
        d_omega = x.flatten() * y.flatten()

        r, g, b = self.env_map[:, :, 0].flatten(), self.env_map[:, :, 1].flatten(), self.env_map[:, :, 2].flatten()

        r *= d_omega
        g *= d_omega
        b *= d_omega
        r, g, b = r.reshape(-1, 1), g.reshape(-1, 1), b.reshape(-1, 1)

        # both of same dimensions having * multiplying corresponding elements
        r_coeff = np.tile(r, sh_order ** 2) * self.sh_base
        g_coeff = np.tile(g, sh_order ** 2) * self.sh_base
        b_coeff = np.tile(b, sh_order ** 2) * self.sh_base

        # Sum over all directions and obtain RGB in basis of sh
        r_coeff, g_coeff, b_coeff = np.sum(r_coeff, 0), np.sum(g_coeff, 0), np.sum(b_coeff, 0)

        self.sh_coeff = np.array([r_coeff, g_coeff, b_coeff])

    def evaluate(self, theta, phi):
        theta = theta * (self.h - 1) / np.pi
        phi = phi * (self.w - 1) / (2 * np.pi)

        theta_idx1 = np.floor(theta).astype(np.int)
        theta_idx2 = np.ceil(theta).astype(np.int)

        phi_idx1 = np.floor(phi).astype(np.int)
        phi_idx2 = np.ceil(phi).astype(np.int)

        wtheta = np.expand_dims(theta - theta_idx1, axis=1)
        wphi = np.expand_dims(phi - phi_idx1, axis=1)

        return ((1 - wphi) * self.env_map[theta_idx1, phi_idx1, :] + wphi * self.env_map[theta_idx1, phi_idx2, :]) * (
                    1 - wtheta) + \
               ((1 - wphi) * self.env_map[theta_idx2, phi_idx1, :] + wphi * self.env_map[theta_idx2, phi_idx2,
                                                                            :]) * wtheta

    def evaluate_gamma(self, theta, phi):
        theta = theta * (self.h - 1) / np.pi
        phi = phi * (self.w - 1) / (2 * np.pi)

        theta_idx1 = np.floor(theta).astype(np.int)
        theta_idx2 = np.ceil(theta).astype(np.int)

        phi_idx1 = np.floor(phi).astype(np.int)
        phi_idx2 = np.ceil(phi).astype(np.int)

        wtheta = np.expand_dims(theta - theta_idx1, axis=1)
        wphi = np.expand_dims(phi - phi_idx1, axis=1)

        return ((1 - wphi) * self.env_map_gamma[theta_idx1, phi_idx1, :] + wphi * self.env_map_gamma[theta_idx1,
                                                                                  phi_idx2, :]) * (1 - wtheta) + \
               ((1 - wphi) * self.env_map_gamma[theta_idx2, phi_idx1, :] + wphi * self.env_map_gamma[theta_idx2,
                                                                                  phi_idx2, :]) * wtheta

    def __save__(self):
        self.__check__()
        np.save(str(self.image_path.parent.parent / 'light_sh_coeffs') + '/' + self.image_path.name.split('.')[
            0] + '_sh_coeffs.npy', self.sh_coeff)

    def __check__(self):
        r, g, b = self.sh_coeff[0].reshape(1, -1), self.sh_coeff[1].reshape(1, -1), self.sh_coeff[2].reshape(1, -1)
        out_r = np.sum(self.sh_base * np.tile(r, (self.h * self.w, 1)), 1)
        out_g = np.sum(self.sh_base * np.tile(g, (self.h * self.w, 1)), 1)
        out_b = np.sum(self.sh_base * np.tile(b, (self.h * self.w, 1)), 1)
        self.back_projection = np.zeros((self.h, self.w, self.c))

        self.back_projection[:, :, 0] = out_r.reshape(self.h, self.w)
        self.back_projection[:, :, 1] = out_g.reshape(self.h, self.w)
        self.back_projection[:, :, 2] = out_b.reshape(self.h, self.w)

        self.back_projection = (self.back_projection.clip(0, 1) * 255).astype('uint8')
