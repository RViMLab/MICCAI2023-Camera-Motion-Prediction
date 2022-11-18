from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
from kornia.geometry.transform import get_perspective_transform
from sklearn.model_selection import train_test_split


def four_point_homography_to_matrix(uv_img: torch.Tensor, duv: torch.Tensor) -> torch.Tensor:
    r"""Transforms homography from four point representation of shape 4x2 to matrix representation of shape 3x3.

    Args:
        uv_img (torch.Tensor): Image edges in image coordinates
        duv (torch.Tensor): Deviation from edges in image coordinates
    
    Return:
        h (torch.Tensor): Homography of shape 3x3.

    Example:

        h = four_point_homography_to_matrix(uv_img, duv)
    """
    uv_wrp = uv_img + duv
    try:
        h = get_perspective_transform(uv_img.flip(-1), uv_wrp.flip(-1))
        return h
    except:
        print("Couldn't solve for homography.")


def integrate_duv(uv: torch.Tensor, duv: torch.Tensor) -> torch.Tensor:
    r"""Sum duv to obtain trajectory.
    
    Args:
        uv (torch.Tensor): Image edges in image coordinates of shape Bx4x2
        duv (torch.Tensor): Deviation from edges in image coordinates of shape Bx4x2
    
    Return:
        uv_int (torch.Tensor): Integrated duv with uv as starting point of shape Bx4x2
    """
    if len(uv.size()) != 3:
        raise ValueError("Expected 3 dimensional input for uv, got {} dimensional.".format(len(uv.size())))
    if len(duv.size()) != 3:
        raise ValueError("Expected 3 dimensional input for duv, got {} dimensional.".format(len(duv.size())))
    return uv + torch.cumsum(duv, dim=0)


def differentiate_duv(duv: torch.Tensor, batch_first: bool=True) -> torch.Tensor:
    r"""Computes the finite difference of duv.

    Args:
        duv (torch.Tensor): Deviation from edges in image coordinates of shape BxTx4x2
        batch_first (bool): If true, expects input of shape BxTx.., else TxBx...

    Return:
        dduv (torch.Tensor): Differentiated duv of shape Bx(T-1)x4x2
    """
    if batch_first:
        dduv = duv.narrow(1, 1, duv.size(1) - 1) - duv.narrow(1, 0, duv.size(1) - 1)
    else:
        dduv = duv.narrow(0, 1, duv.size(0) - 1) - duv.narrow(0, 0, duv.size(0) - 1)
    return dduv


def image_edges(img: torch.Tensor) -> torch.Tensor:
    r"""Returns edges of image (uv) in OpenCV convention.

    Args:
        img (torch.Tensor): Image of shape BxCxHxW

    Return:
        uv (torch.Tensor): Image edges of shape Bx4x2
    """
    if len(img.shape) != 4:
        raise ValueError("Expected 4 dimensional input, got {} dimensions.".format(len(img.shape)))
    shape = img.shape[-2:]
    uv = torch.tensor(
        [
            [       0,        0],
            [       0, shape[1]],
            [shape[0], shape[1]],
            [shape[0],        0]
        ], device=img.device, dtype=torch.float32
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


def unique_video_train_test(df: pd.DataFrame, train_split: float=0.8, tolerance: float=0.05, random_state: int=42):
    r"""Splits videos into train and test set.

    Args:
        df (pd.DataFrame): Pandas dataframe, must contain {'folder': , 'file': , 'vid': , 'frame': }
        train_split (float): Fraction of train data, default 0.8
        tolerance (float): Split tolerance, train_split + tolerance <= len(df[df.train] == False)/len(df) <= train_split + tolerance
        random_state (int): Random state for deterministic splitting
    
    Return:
        df (pd.DataFrame): Pandas dataframe, contain {'folder': , 'file': , 'vid': , 'frame': , 'train': }
    """
    # find unique videos
    unique_vid = df.vid.unique()

    _, test_vid = train_test_split(
        unique_vid,
        train_size=train_split,
        random_state=random_state
    )

    df['train'] = True
    df.loc[df.vid.isin(test_vid), 'train'] = False

    # assert if fraction off
    fraction = len(df[df.train == False])/len(df)
    assert np.isclose(
        fraction, 1 - train_split, atol=tolerance
    ), 'Train set fraction {:.3f} not close enough to (1 - train_split) {} at tolerance {}'.format(fraction, 1 - train_split, tolerance)

    return df


if __name__ == '__main__':

    def test_forward_backward_sequence():
        import numpy as np
        import torch

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

    def test_differentiate_duv():
        import torch

        B = 2
        T = 10
        duv = torch.ones([B,T,4,2])
        dduv = differentiate_duv(duv, True)
        if dduv.shape[1] != T-1:
            raise ValueError("Shape of dduv must be T-1.")
        if dduv.nonzero().nelement() != 0:
            raise ValueError("Expected derivative to equal zero.")
        duv = torch.ones([T,B,4,2])
        dduv = differentiate_duv(duv, False)
        if dduv.shape[0] != T-1:
            raise ValueError("Shape of dduv must be T-1.")
        if dduv.nonzero().nelement() != 0:
            raise ValueError("Expected derivative to equal zero.")

    test_differentiate_duv()

