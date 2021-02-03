import numpy as np
from typing import List


class EndoscopyCircle(object):
    @staticmethod
    def randomCenter(shape: tuple, c_off_scale: float=0.125, seed: np.int32=None) -> tuple:
        r"""Helper function to generate random center.

        Args:
            shape (tuple): Image shape HxWxC
            c_off_scale (float): Center offset scale of image shape. Perturbes endoscopic view around image center
            seed (np.int32): Seed for deterministic output

        Return:
            center (tuple): Center of endoscopic view
        """
        np.random.seed(seed)
        offset = (np.random.rand(2)*2 - 1)*shape[:-1]*c_off_scale
        center = (shape[1]/2 + offset[1], shape[0]/2 + offset[0])
        np.random.seed(None)

        return center

    @staticmethod
    def randomCenterUpdate(shape: tuple, center: tuple, scale: float=0.1, chance: float=0.1, seed: np.int32=None) -> tuple:
        r"""Helper function to randomly update center.

        Args:
            shape (tuple): Image shape HxWxC
            center (tuple): Initial endoscopic view center
            scale (float): Center update scale
            chance (float): Chance by which center is updated
            seed (np.int32): Seed for deterministic output

        Return:
            new_center (tuple): Updated center
        """
        np.random.seed(seed)
        if np.random.rand(1) <= chance:
            new_center = (
                center[0] + shape[1]*(np.random.rand(1)*2. - 1.)*scale,
                center[1] + shape[0]*(np.random.rand(1)*2. - 1.)*scale
            )
            np.random.seed(None)
            return new_center
        else:
            np.random.seed(None)
            return center

    @staticmethod
    def randomRadius(shape: tuple, r_min_scale: float=0.25, r_amp_scale: float=0.5, seed: np.int32=None) -> float:
        r"""Helper function to generate random radius. Circle's radius is computed as 
        r_min_scale + [0, 1)*r_amp_scale, where r_min_scale scales the minimum image size, and r_amp_scale scales the maximum image size.

        Args:
            shape (tuple): Image shape HxWxC
            r_min_scale (float): Minimum radius, scale of minimum image size
            r_amp_scale (float): Radius amplitude, scale of maximum image size
            seed (np.int32): Seed for deterministic output

        Return:
            radius (float): Radius of endoscopic view
        """
        np.random.seed(seed)
        r_min_pix = r_min_scale*min(shape[:-1])
        r_amp_pix = r_amp_scale*max(shape[:-1])
        radius = r_min_pix + np.random.rand(1)*r_amp_pix
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

    ec = EndoscopyCircle()

    img = np.load('utils/transforms/sample.npy')

    # sample 10 loops
    for _ in range(100):
        center = EndoscopyCircle.randomCenter(img.shape)
        radius = EndoscopyCircle.randomRadius(img.shape, seed=5)  # seed = np.random.randint(np.iinfo(np.int32).max) # set random seed for numpy

        img0 = ec(img, center=center, radius=radius)

        radius = ec.randomRadius(img.shape)

        img1 = ec(img, center=center, radius=radius)

        # random update
        center = EndoscopyCircle.randomCenterUpdate(img.shape, center)
        img2 = ec(img, center=center, radius=radius)

        cv2.imshow('img0', img0)
        cv2.imshow('img1', img1)
        cv2.imshow('img2', img2)
        cv2.waitKey()
