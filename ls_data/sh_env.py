import os
import cv2
from imageio.v2 import imread,imwrite
from argparse import ArgumentParser
import numpy as np
from scipy.special import sph_harm
import pyshtools
# from pyshtools import shtools

from envmap import EnvironmentMap

# Sanity check, sph_harm was bogus in some versions of scipy / Anaconda Python
# http://stackoverflow.com/questions/33149777/scipy-spherical-harmonics-imaginary-part
#assert np.isclose(sph_harm(2, 5, 2.1, 0.4), -0.17931012976432356-0.31877392205957022j), \
#    "Please update your SciPy version, the current version has a bug in its " \
#    "spherical harmonics basis computation."


class SphericalHarmonic:
    def __init__(self, input_, copy_=True, max_l=None, norm=4):
        """
        Projects `input_` to its spherical harmonics basis up to degree `max_l`.
        
        norm = 4 means orthonormal harmonics.
        For more details, please see https://shtools.oca.eu/shtools/pyshexpanddh.html
        """

        if copy_:
            self.spatial = input_.copy()
        else:
            self.spatial = input_

        # if not isinstance(self.spatial, EnvironmentMap):
        #     self.spatial = EnvironmentMap(self.spatial, 'LatLong')

        # if self.spatial.format_ != "latlong":
        #     self.spatial = self.spatial.convertTo("latlong")

        self.norm = norm
        
        self.coeffs = []
        for i in range(self.spatial.data.shape[2]):
            self.coeffs.append(pyshtools.expand.SHExpandDH(self.spatial[:,:,i], norm=norm, sampling=2, lmax_calc=max_l))

    def reconstruct(self, height=None, max_l=None, clamp_negative=True):
        """
        :height: height of the reconstructed image
        :clamp_negative: Remove reconstructed values under 0
        """

        retval = []
        for i in range(len(self.coeffs)):
            retval.append(pyshtools.expand.MakeGridDH(self.coeffs[i], norm=self.norm, sampling=2, lmax=height, lmax_calc=max_l))

        retval = np.asarray(retval).transpose((1,2,0))

        if clamp_negative:
            retval = np.maximum(retval, 0)

        return retval
    
    def window(self, function="sinc"):
        """
        Applies a windowing function to the coefficients to reduce ringing artifacts.
        See https://www.ppsloan.org/publication/StupidSH36.pdf
        """
        deg = self.coeffs[0].shape[2]
        x = np.linspace(0, 1, deg + 1)[:-1]

        if function == "sinc":
            kernel = np.sinc(x)
        else:
            raise NotImplementedError(f"Windowing function {function} is not implemented.")
    
        for c in range(len(self.coeffs)):
            for l in range(self.coeffs[c].shape[2]):
                self.coeffs[c][0,l,:] *= kernel   # real
                self.coeffs[c][1,l,:] *= kernel   # imag

        return self

def parse_args():
    parser =  ArgumentParser(description="project envmap to SH")
    parser.add_argument("--input_light", default="", help="specify calib file location")
    args = parser.parse_args()
    return args
def read_image(filename):
    """Read the image file and return its contents as a numpy array"""
    image = imread(filename)
    h,w,c = image.shape
    nw,nh = int(w/4),int(h/4)
    # nw,nh = 64,64
    image = cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA)
    return image

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    args = parse_args()
    # e = EnvironmentMap(args.input_light, 'angular')
    # e.resize((64, 64))
    # e.convertTo('latlong')
    env_map = read_image(args.input_light)
    se = SphericalHarmonic(env_map)
    print(se.coeffs[0].shape)
    recons = se.reconstruct(height=32)
    print(recons.shape)
    imwrite('env_map.exr',env_map.astype('float32'))
    imwrite('recons.exr',recons.astype('float32'))

    