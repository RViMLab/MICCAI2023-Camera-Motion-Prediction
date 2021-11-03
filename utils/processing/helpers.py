import numpy as np
import torch
from typing import Tuple, Union
from kornia.geometry.transform import get_perspective_transform


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


def forward_backward_sequence(video: Union[np.ndarray, torch.Tensor], step: int=2, last_step: int=1) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
    r"""Helper function to sample unique forward-backward images from image sequence.

    Args:
        video (torch.Tensor): Image sequence of shape BxNxCxHxW, best to have even sequence length if step=2 and last_step=1, e.g. 6, 4
        step (int): Step between frames
        last_step (int): Last step in sequence

    Return:
        forward (torch.Tensor): Forward image sequence
        backward (troch.Tensor): Backward image sequence
    """
    if isinstance(video, torch.Tensor):
        forward  = video[:,:-last_step:step]
        backward = video.flip(1)[:,:-last_step:step]  # flip video and run forward

        forward  = torch.cat((forward, backward[:,0].unsqueeze(1)), axis=1)
        backward = torch.cat((backward, forward[:,0].unsqueeze(1)), axis=1)
    elif isinstance(video, np.ndarray):
        forward  = video[:,:-last_step:step]
        backward = video[:,::-1][:,:-last_step:step]  # flip video and run forward

        forward  = np.concatenate((forward, np.expand_dims(backward[:,0],1)), axis=1)
        backward = np.concatenate((backward, np.expand_dims(forward[:,0],1)), axis=1)
    else:
        raise ValueError('Unsupported type: {}'.format(type(video)))

    return forward, backward


if __name__ == '__main__':
    import torch
    import numpy as np

    seq_len = 3
    step = 2
    last_step = 1

    # np
    dummy_vid = np.zeros([1,seq_len,1,2,2])
    for i in range(seq_len):
        dummy_vid[:, i] = i

    forward, backward = forward_backward_sequence(dummy_vid, step, last_step)
    print(forward)
    print(backward)

    # torch
    dummy_vid = torch.zeros([1,seq_len,1,2,2])
    for i in range(seq_len):
        dummy_vid[:,i] = i

    forward, backward = forward_backward_sequence(dummy_vid, step, last_step)
    print(forward)
    print(backward)
