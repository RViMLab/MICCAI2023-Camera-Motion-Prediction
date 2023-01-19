from typing import Union

import numpy as np
import torch
from kornia.geometry import warp_perspective

from ..processing import four_point_homography_to_matrix, image_edges


def yt_alpha_blend(
    img_y: Union[np.ndarray, torch.Tensor],
    img_t: Union[np.ndarray, torch.Tensor],
    alpha: float = 0.5,
) -> Union[np.ndarray, torch.Tensor]:
    r"""Blends RGB image into yellow and turquoise.

    Args:
        img_y (np.ndarray or torch.Tensor): Image to be blended in yellow color (np.ndarray: CxHxW, torch.Tensor: ...xHxWxC)
        img_t (np.ndarray or torch.Tensor): Image to be blended in turquoise color (np.ndarray: CxHxW, torch.Tensor: ...xHxWxC)

    Returns:
        blend (np.ndarray or torch.Tensor): Blend of the form alpha*img_y + (1-alpha)*img_t
    """
    if type(img_y) == np.ndarray and type(img_t) == np.ndarray:
        img_y_cpy = img_y.copy()
        img_t_cpy = img_t.copy()

        img_y_cpy[..., 0] = 0
        img_t_cpy[..., -1] = 0
    elif type(img_y) == torch.Tensor and type(img_t) == torch.Tensor:
        img_y_cpy = img_y.detach().clone()
        img_t_cpy = img_t.detach().clone()

        img_y_cpy[..., 0, :, :] = 0
        img_t_cpy[..., -1, :, :] = 0

    blend = alpha * img_y_cpy + (1 - alpha) * img_t_cpy
    return blend


def create_blend_from_four_point_homography(
    frames_i: torch.Tensor, frames_ips: torch.Tensor, duvs: torch.Tensor
) -> torch.Tensor:
    r"""Helper function that creates blend figure, given four point homgraphy representation.

    Args:
        frames_i (torch.Tensor): Frames i of shape NxCxHxW
        frames_ips (torch.Tensor): Frames i+step of shape NxCxHxW
        duvs (torch.Tensor): Edge delta from frames i+step to frames i of shape Nx4x2

    Return:
        blends (torch.Tensor): Blends of warp(frames_i) and frames_ips
    """
    uvs = image_edges(frames_i)
    Hs = four_point_homography_to_matrix(uvs, duvs)
    try:  # handle inversion error
        wrps = warp_perspective(frames_i, torch.inverse(Hs), frames_i.shape[-2:])
        blends = yt_alpha_blend(frames_ips, wrps)
    except:
        return frames_i
    return blends
