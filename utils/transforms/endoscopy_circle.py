import numpy as np
from typing import List


class EndoscopyCircle(object):
    def __init__(self, c_off: float=0.125, r_min: float=0.25, r_amp: float=0.5, seeds: np.int32=None):
        r"""Generates endoscopic circular view with random noise as padding.
        Circles radius is computed as r_min + [0, 1)*r_amp, where r_min scales the
        minimum image size, and r_amp scales the maximum image size.

        Args:
            c_off (float): Center offset scale of image shape. Perturbes endoscopic view around image center
            r_min (float): Minimum radius, scale of minimum image size
            r_amp (float): Radius amplitude, scale of maximum image size
        """
        self._c_off = c_off
        self._r_min = r_min
        self._r_amp = r_amp

    def randomCenter(self, shape, , seed):
        r"""Helper function to generate random center.

        Args:
            seed (np.int32): Seed for deterministic output

        Return:
            center (tuple): Center of endoscopic view
        """


    def randomRadius(self, shape, r_min, r_amp, seed: np.int32=None) -> float:
        r"""Helper function to generate random radius.

        Args:
            seed (np.int32): Seed for deterministic output

        Return:
            radius (float): Radius of endoscopic view
        """
        np.random.seed(seed)
        r_min = self._r_min*min(img.shape[:-1])
        r_amp = self._r_amp*max(img.shape[:-1])
        radius = r_min + np.random.rand(1)*r_amp
        np.random.seed(None)

        return radius

    def __call__(self, img: np.array, center: tuple=None, radius: float=None):
        r"""Puts a circular noisy mask on top of an image.

        Args:
            img (np.array): uint8 image to be masked of shape HxWxC
            center (tuple): Circular mask's center
            radius (float): Circular mask's radius
        """
        return self._circularNoisyMask(img, center, radius)

    def _circularNoisyMask(self, img: np.array, center: tuple=None, radius: float=None) -> np.array:
        r"""Puts a circular noisy mask on top of an image, mask from https://stackoverflow.com/questions/17394882/how-can-i-add-new-dimensions-to-a-numpy-array.

        Args:
            img (np.array): uint8 image to be masked of shape HxWxC
            center (tuple): Circular mask's center
            radius (float): Circular mask's radius

        Return:
            img (np.array): Masked image
        """
        h, w = img.shape[: -1]

        if center is None: # use the middle of the image
            center = (int(w/2), int(h/2))
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = np.expand_dims(np.where(dist_from_center <= radius, 1, 0), axis=-1).astype(np.uint8)
        img = mask*img + ((1-mask)*np.random.random(img.shape)*255).astype(np.uint8)

        return img

if __name__ == '__main__':
    import cv2

    ec = EndoscopyCircle(0.25, 0.4)

    img = np.load('utils/transforms/sample.npy')

    # sample 10 loops
    for _ in range(100):
        offset = (np.random.rand(2)*2 - 1)*img.shape[:-1]/8
        center = (img.shape[1]/2 + offset[1], img.shape[0]/2 + offset[0])

        seed = 5  # seed = np.random.randint(np.iinfo(np.int32).max) # set random seed for numpy
        
        img0 = ec(img, center=center, seed=seed)
        img1 = ec(img, center=center)

        cv2.imshow('img0', img0)
        cv2.imshow('img1', img1)
        cv2.waitKey()
