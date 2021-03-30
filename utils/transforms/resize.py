import cv2
import numpy as np


# interpolation flags
# https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121
INTERPOLATION_DICT = {
    'nearest': cv2.INTER_NEAREST,
    'linear': cv2.INTER_LINEAR,
    'cubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos4': cv2.INTER_LANCZOS4
}


class Resize(object):
    def __init__(self, dsize: list, interpolation: str='cubic'):
        r"""Callable resize operation.

        Args:
            dsize (list of int): OpenCV order, (W, H)
            interpolation (str): Interpolation type - nearest, linear, cubic, area, lanczos4
        """
        self.dsize = tuple(dsize)
        self.interpolation = INTERPOLATION_DICT[interpolation]

    def __call__(self, img: np.array):
        return cv2.resize(img, dsize=self.dsize, interpolation=self.interpolation)


if __name__ == '__main__':
    ones = np.ones([100, 200])
    resize = Resize([100, 50])
    ones = resize(ones)
    print(ones.shape)
