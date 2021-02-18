import numpy as np
from typing import List, Tuple, Callable
import torch
from torch.utils.data import Dataset
from torchvision.datasets.video_utils import VideoClips


class VideoDataset(Dataset):
    def __init__(self, video_paths: List[str], clip_length_in_frames: int=25, frames_between_clips: int=1, precomputed_metadata: dict=None, num_workers: int=0, transforms: List[Callable]=None, seeds: bool=False) -> None:
        r"""Dataset to load video clips with homographies

        Args:
            video_paths (List[str]): List of paths to video files
            clip_length_in_frames (int): Preview horizon, frames per returned clip
            frames_between_clips (int): Offset frames between starting point of clips
            precomputed_metadata (dict): Metadata
            num_worker (int): Number of subprocesses for loading images from video
            transforms (List[Callable]): List of callable tranforms (video specific transforms)
            seeds (bool): Seeds for deterministic output, e.g. for test set
        """
        if transforms is not None and len(video_paths) != len(transforms): 
            raise ValueError("Length of provided videos paths must equal length of provided transforms.")

        self._video_clips = VideoClips(
            video_paths=video_paths,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            _precomputed_metadata=precomputed_metadata,
            num_workers=num_workers
        )
        self._transforms = transforms
        self._seeds = seeds

    @property
    def metadata(self):
        return self._video_clips.metadata

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        video, audio, info, video_idx = self._video_clips.get_clip(idx)

        video = video.permute(0, 3, 1, 2).contiguous()  # NxHxWxC -> NxCxHxW
        # video = video.float()/255. # uint8 [0, 255] -> float [0., 1.] <- times out

        # set seed if desired
        if self._seeds:
            seed = idx
        else:
            seed = np.random.randint(np.iinfo(np.int32).max)  # set random seed for numpy

        if self._transforms:
            torch.manual_seed(seed)
            video = self._transforms[video_idx](video)

        return video 

    def __len__(self):
        return self._video_clips.num_clips()
