import numpy as np


class Crop(object):
    def __init__(self, top_left_corner: list, shape: list):
        r"""Callable crop operation.

        Args:
            top_left_corner (list of int): top_left_corner of crop
            shape (list of int): Shape of crop, HxW
        """
        self.top_left_corner = top_left_corner
        self.shape = shape

    def __call__(self, img: np.array):
        r"""Crop call.

        Args:
            img (np.array): Expects array of shape HxWxC
        """
        return img[
            self.top_left_corner[0]:self.top_left_corner[0]+self.shape[0], 
            self.top_left_corner[1]:self.top_left_corner[1]+self.shape[1]
        ]


if __name__ == '__main__':
    ones = np.ones([100, 200])
    crop = Crop(top_left_corner=[10, 10], shape=[50, 50])
    ones = crop(ones)
    print(ones.shape)
