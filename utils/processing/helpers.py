import torch
from kornia import get_perspective_transform


def fourPtToMatrixHomographyRepresentation(uv_img: torch.Tensor, duv: torch.Tensor):
    r"""Transforms homography from four point representation of shape 4x2 to matrix representation of shape 3x3.

    Args:
        uv_img (torch.Tensor): Image edges in image coordinates
        duv (torch.Tensor): Deviation from edges in image coordinates

    Example:

        h = fourPtToMatrixHomographyRepresentation(uv_img, duv)
    """
    uv_wrp = uv_img + duv
    h = get_perspective_transform(uv_img.flip(-1), uv_wrp.flip(-1))
    return h


def imageEdges(img: torch.Tensor):
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
