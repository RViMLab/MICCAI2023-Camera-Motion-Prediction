import numpy as np
import torch
from typing import Tuple, Union


class Crop(object):
    def __init__(self, top_left_corner: np.array, shape: Tuple[int], order: str='hwc'):
        r"""Callable crop operation.

        Args:
            top_left_corner (np.array): top_left_corner of crop
            shape (tuple of int): Shape of crop, HxW
            order (str): Channel order, e.g. 'hwc'
        """
        self._top_left_corner = top_left_corner
        self._shape = shape
        self._order = order

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        r"""Crop call.

        Args:
            img (np.array): Expects array of shape self._order
        """
        if self._order == 'hwc':
            return img[
                self._top_left_corner[0]:self._top_left_corner[0]+self._shape[0], 
                self._top_left_corner[1]:self._top_left_corner[1]+self._shape[1]
            ]
        elif self._order == 'chw':
            return img[...,
                self._top_left_corner[0]:self._top_left_corner[0]+self._shape[0], 
                self._top_left_corner[1]:self._top_left_corner[1]+self._shape[1]
            ]
        else:
            raise ValueError('Unknown order: {}'.format(self._order))


if __name__ == '__main__':
    ones = np.ones([100, 200])
    crop = Crop(top_left_corner=[10, 10], shape=[50, 50])
    ones = crop(ones)
    print(ones.shape)

    ones = torch.ones([1, 1, 100, 200])
    crop = Crop(top_left_corner=[10, 10], shape=[50, 50], order='chw')
    ones = crop(ones)
    print(ones.shape)
