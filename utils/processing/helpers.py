import torch
from typing import Tuple
from kornia import get_perspective_transform


def four_point_homography_to_matrix(uv_img: torch.Tensor, duv: torch.Tensor) -> torch.Tensor:
    r"""Transforms homography from four point representation of shape 4x2 to matrix representation of shape 3x3.

    Args:
        uv_img (torch.Tensor): Image edges in image coordinates
        duv (torch.Tensor): Deviation from edges in image coordinates

    Example:

        h = four_point_homography_to_matrix(uv_img, duv)
    """
    uv_wrp = uv_img + duv
    h = get_perspective_transform(uv_img.flip(-1), uv_wrp.flip(-1))
    return h


def image_edges(img: torch.Tensor) -> torch.Tensor:
    r"""Returns edges of image (uv) in OpenCV convention.

    Args:
        img (torch.Tensor): Image of shape BxCxHxW

    Returns:
        uv (torch.Tensor): Image edges of shape 1x4x2
    """
    shape = img.shape[-2:]
    uv = torch.tensor(
        [
            [       0,        0],
            [       0, shape[1]],
            [shape[0], shape[1]],
            [shape[0],        0]
        ], device=img.device, dtype=img.dtype
    )
    return uv.unsqueeze(0).repeat(img.shape[0], 1, 1)


def frame_pairs(video: torch.Tensor, step: int=1) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Helper function to return frame pairs at an offset.

    Args:
        video (torch.Tensor): Video clip of shape BxNxCxHxW
        step (int): Number of frames in between image pairs

    Return:
        frames_i (torch.Tensor): Frames starting at time step i with stride step
        frames_ips (torch.Tensor): Frames starting at time step i+step with stride step
    """
    frames_i   = video[:,:-step:step]
    frames_ips = video[:,step::step]
    return frames_i, frames_ips

