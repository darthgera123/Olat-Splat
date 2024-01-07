# -*- coding: utf-8 -*-
# @File    : legendre_sh.py

import math
import torch
import numpy as np
from sh_utils import *


"""
Refer to https://www.ppsloan.org/publications/SHJCGT.pdf
"""

factorials_values = torch.Tensor(
    [
        1.0,
        1.0,
        2.0,
        6.0,
        24.0,
        120.0,
        720.0,
        5040.0,
        40320.0,
        362880.0,
        3628800.0,
        39916800.0,
        479001600.0,
        6227020800.0,
        87178291200.0,
        1307674368000.0,
        20922789888000.0,
        355687428096000.0,
        6402373705728000.0,
        121645100408832000.0,
        2432902008176640000.0,
        51090942171709440000.0,
        1124000727777607680000.0,
        25852016738884976640000.0,
        620448401733239439360000.0,
        15511210043330985984000000.0,
        403291461126605635584000000.0,
        10888869450418352160768000000.0,
        304888344611713860501504000000.0,
        8841761993739701954543616000000.0,
        265252859812191058636308480000000.0,
        8222838654177922817725562880000000.0,
        263130836933693530167218012160000000.0,
        8683317618811886495518194401280000000.0
    ]
)

sqrt_2 = 1.4142135623730951

double_factorial = torch.Tensor(
    [
        1.0,
        1.0,
        3.0000000000000004,
        15.000000000000004,
        105.00000000000001,
        945.0,
        10395.0,
        135135.00000000003,
        2027025.0000000002,
        34459425.0,
        654729075.0000001,
        13749310575.000004,
        316234143225.00006,
        7905853580625.002,
        213458046676875.03,
        6190283353629376.0,
        1.918987839625107e+17,
        6.332659870762852e+18,
        2.2164309547669976e+20,
        8.200794532637893e+21,
        3.1983098677287775e+23,
        1.3113070457687992e+25,
        5.638620296805837e+26,
        2.5373791335626256e+28,
        1.1925681927744346e+30,
        5.843584144594731e+31,
        2.980227913743312e+33,
        1.5795207942839554e+35,
        8.687364368561751e+36,
        4.9517976900801995e+38
    ]
)

def P(l: int, m: int, cos_theta: np.array):
    """
    Computes P_l^{l} term
    :param l: order value
    :param m: Positive value is expected
    :param cos_theta:
    :return:
    """
    device = 'cuda'
    pmm = torch.ones((cos_theta.shape[0])).to(device)
    if m > 0:
        sin_theta = torch.sqrt(1 - torch.pow(cos_theta, 2))
        pmm = torch.pow(sin_theta, m) * double_factorial[m]
        pmm = -pmm if m & 1 == 1 else pmm
        # fact = 1.
        # for i in range(1, m + 1):
        #     pmm = pmm * (-fact) * sin_theta
        #     fact += 2.

    if l == m:
        return pmm

    pmmp1 = (2 * m + 1.) * cos_theta * pmm
    if l == m + 1:
        return pmmp1

    pll = torch.zeros((cos_theta.shape[0])).to(device)

    for ll in range(m + 2, l + 1):
        pll = ((2 * ll - 1.) * cos_theta * pmmp1 - (ll + m - 1.) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll


def K(l: int, m: int):
    """
    Return K factor in SH Coefficient Calculation
    :param l: Order
    :param m: -l to l values
    :return: K
    """
    # m is always supposed to be positive
    return math.sqrt((2 * l + 1) * factorials_values[l - m] / (4 * math.pi * factorials_values[l + m]))


def compute_sh(theta, phi, SH_ORDER=5):
    """
    Compute SH Coefficients  Required the theta and phi Spherical Angles
    :param theta: Angle made in z direction
    :param phi: For XY plane
    :param SH_ORDER: Value of l
    :return: Sh Coefficients (N-directions X SH_ORDER*SH_ORDER)
    """
    cos_theta = torch.cos(theta)
    coefficients = []
    for l in range(SH_ORDER):
        m_values = list(range(-l, l+1))
        for m in m_values:
            if m == 0:
                coefficients += [K(l, m) * P(l, m, cos_theta)]
            elif m > 0:
                coefficients += [sqrt_2 * K(l, m) * torch.cos(m * phi) * P(l, m, cos_theta)]
            else:
                coefficients += [sqrt_2 * K(l, -m) * torch.sin(-m * phi) * P(l, -m, cos_theta)]

    sh_coeffs = torch.stack(coefficients)
    return sh_coeffs.T
