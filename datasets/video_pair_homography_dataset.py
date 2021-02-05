import pandas as pd
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets.video_utils import VideoClips

from utils import recursiveMethodCallFromDictList


class VideoPairHomographyDataset(Dataset):
    def __init__(self, video_paths: List[str], H_df: pd.DataFrame=None, clip_length_in_frames: int=25, frames_between_clips: int=1, transforms: List[dict]=None) -> None:
        r"""Dataset to load video clips with homographies

        Args:
            video_paths (List[str]): List of paths to video files
            H_df (pandas.DataFrame): Dataframe with homographies for consecutive frames
            clip_length_in_frames (int): Preview horizon, frames per returned clip
            frames_between_clips (int): Offset frames between starting point of clips
            transforms (List[dict]): List of dictionaries, e.g. [{'resize': {'shape': [10, 10]}}] (transforms from torchvision.transforms.functional)
        """
        if len(video_paths) != len(transforms): 
            raise ValueError("Length of provided videos paths must equal length of provided transforms.")

        self._video_clips = VideoClips(
            video_paths=video_paths,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips
        )
        self._H_df = H_df
        self._transforms = transforms

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        video, audio, info, video_idx = self._video_clips.get_clip(idx)

        video = video.permute(0, 3, 1, 2)  # NxHxWxC -> NxCxHxW 

        if self._transforms:
            video = recursiveMethodCallFromDictList(video, self._transforms[video_idx], torchvision.transforms.functional)
        
        if self._H_df:
            return video, 
        else:
            return video, None
        

    def __len__(self):
        return self._video_clips.num_clips()

    @staticmethod
    def framePairs(video: torch.Tensor, step: int=1) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Helper function to return frame pairs at an offset.

        Args:
            video (torch.Tensor): Video clip of shape NxCxHxW
            step (int): Number of frames in between image pairs

        Return:
            frames_i (torch.Tensor): Frames starting at time step i with stride step
            frames_ips (torch.Tensor): Frames starting at time step i+step with stride step
        """
        frames_i   = video[:-step:step]
        frames_ips = video[step::step]
        return frames_i, frames_ips

    @staticmethod
    def generateHomographyDataframe() -> pd.DataFrame:
        r"""Helper function to generate a homography dataframe.

        Args:

        Return:
            H_df (pandas.DataFrame): Dataframe holding homographies for consecutive frames {'H': np.array, 'vid_idx': int, 'frame_idcs': list}
        """
        # paths etc
        # loop over videos
        # safe to homography dataframe

