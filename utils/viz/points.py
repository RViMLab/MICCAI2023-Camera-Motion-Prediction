import cv2
import numpy as np
import torch
from typing import Union, Tuple
from kornia import tensor_to_image


def draw_points(img: Union[torch.Tensor, np.ndarray], pts: Union[torch.Tensor, np.ndarray], rad: int=1, col: Tuple[int]=(255, 0, 255)) -> np.ndarray:
    r"""Draw points into an image. Points follow OpenCV convention (x, y) <-> (1, 0).

    Args:
        img (Union[torch.Tensor, np.ndarray]): Image to draw points into of shape CxHxW, HxWxC
        pts (Union[torch.Tensor, np.ndarray]): Points to draw of shape BxNx2
        rad (int): Radius of points
        col (Tuple[int]): Color

    Return:
        img (Union [torch.Tensor, np.ndarray]): Image with points
    """
    if isinstance(img, torch.Tensor):
        img = tensor_to_image(img).copy()

    for pt in pts:
        img = cv2.circle(img, (int(pt[0]), int(pt[1])), radius=rad, color=col, thickness=cv2.FILLED)
    
    return img


if __name__ == '__main__':
    img = np.ones([255, 255, 3])
    pts = np.array([
        [10, 20],
        [10, 30]
    ])

    img = draw_points(img, pts, rad=2, col=(255,0,0))

    cv2.imshow('img', img)
    cv2.waitKey()

    img = torch.ones([3, 255, 255])

    img = draw_points(img, pts, rad=2, col=(255,0,0))

    cv2.imshow('img', img)
    cv2.waitKey()
