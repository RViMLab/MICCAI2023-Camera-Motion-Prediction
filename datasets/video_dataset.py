import random
import numpy as np
from typing import List, Tuple, Callable
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ConvertImageDtype
from torchvision.datasets.video_utils import VideoClips


class VideoDataset(Dataset):
    def __init__(self, 
        video_paths: List[str], 
        clip_length_in_frames: int=25, 
        frames_between_clips: int=1, 
        frame_rate: int=1, 
        precomputed_metadata: dict=None, 
        num_workers: int=0, 
        pre_transforms: List[Callable]=None, 
        aug_transforms: List[Callable]=None, 
        seeds: bool=False,
        permute: bool=True,
        convert_dtype: bool=True
    ) -> None:
        r"""Dataset to load video clips with homographies

        Args:
            video_paths (List[str]): List of paths to video files
            clip_length_in_frames (int): Preview horizon, frames per returned clip
            frames_between_clips (int): Offset frames between starting point of clips
            frame_rate (int): Resampling frame rate
            precomputed_metadata (dict): Metadata
            num_worker (int): Number of subprocesses for loading images from video
            pre_transforms (List[Callable]): List of callable tranforms for cropping an resizing (video specific transforms)
            aug_transforms (List[Callable]): List of callable tranforms for augmentation (video specific transforms)
            seeds (bool): Seeds for deterministic output, e.g. for test set
            permute (bool): Permute sampled sequence channels NxHxWxC -> NxCxHxW (memory intensive)
            convert_dtype (bool): Permute sampled sequence dtype to torch.float32 (memory intensive)
        """
        if pre_transforms is not None and len(video_paths) != len(pre_transforms): 
            raise ValueError("Length of provided videos paths must equal length of provided transforms.")

        self._video_clips = VideoClips(
            video_paths=video_paths,
            clip_length_in_frames=clip_length_in_frames,
            frames_between_clips=frames_between_clips,
            frame_rate=frame_rate,
            _precomputed_metadata=precomputed_metadata,
            num_workers=num_workers
        )
        if pre_transforms is None:
            self._pre_transforms = [None]*(len(video_paths))
        else:
            self._pre_transforms = pre_transforms
        if aug_transforms is None:
            self._aug_transforms = [None]*(len(video_paths))
        else:
            self._aug_transforms = aug_transforms
        self._seeds = seeds
        self._permute = permute

        self._convert_dtype = convert_dtype
        if self._convert_dtype:
            self._dtype_trafo = ConvertImageDtype(torch.float32)

    @property
    def metadata(self):
        return self._video_clips.metadata

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        video, audio, info, video_idx = self._video_clips.get_clip(idx)

        if self._permute:
            video = video.permute(0, 3, 1, 2).contiguous()  # NxHxWxC -> NxCxHxW

        # crop and resize video
        if self._pre_transforms[video_idx]:
            video = self._pre_transforms[video_idx](video)

        # set seed if desired
        if self._seeds:
            seed = idx
        else:
            seed = random.randint(0, np.iinfo(np.int32).max)  # set random seed for numpy

        augmented_video = torch.empty_like(video)

        if self._aug_transforms[video_idx]:
            augmented_video = video.clone()
            torch.manual_seed(seed)
            augmented_video = self._aug_transforms[video_idx](augmented_video)
            # convert dtype and normalize -> [0., 1.]
            if self._convert_dtype:
                augmented_video = self._dtype_trafo(augmented_video)

        # convert dtype and normalize -> [0., 1.]
        if self._convert_dtype:
            video = self._dtype_trafo(video)

        return video, augmented_video, self._video_clips.frame_rate, self._video_clips.video_fps, video_idx, idx

    def __len__(self):
        return self._video_clips.num_clips()
