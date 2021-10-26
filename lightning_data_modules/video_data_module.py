import os
import pytorch_lightning as pl
from typing import Tuple
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from datasets import VideoDataset
from utils.transforms import anyDictListToCompose


class VideoDataModule(pl.LightningDataModule):
    def __init__(self, 
            meta_df: pd.DataFrame, 
            prefix: str, 
            clip_length_in_frames: int=25, 
            frames_between_clips: int=1, 
            frame_rate: int=1, 
            train_split: float=0.8,
            batch_size: int=32, 
            num_workers: int=2, 
            random_state: int=42, 
            train_metadata: dict=None, 
            val_metadata: dict=None, 
            test_metadata: dict=None
        ) -> None:
        r"""Pytorch Lightning datamodule for videos.
        
        Args:
            meta_df (pd.DataFrame): Meta dataframe containing {'database':, 'train':, 'file': {'name':, 'path':}, 'pre_transforms': [], 'aug_transforms': [], 'auxiliary':}
            prefix (str): Path to the database from meta_df
            clip_length_in_frames (int): Preview horizon, frames per returned clip
            frames_between_clips (int): Offset frames between starting point of clips
            frame_rate (int): Resampling frame rate
            train_slit (float): Relative size of train split
            batch_size (int): Batch size for the dataloader
            num_workers (int): Number of workers for the VideoClip init and for the Dataloader
            random_state (int): Initial random state for the train/validation split
        """
        super().__init__()
        self._meta_df = meta_df
        self._prefix = prefix

        self._clip_length_in_frames = clip_length_in_frames
        self._frames_between_clips = frames_between_clips
        self._frame_rate = frame_rate

        self._train_split = train_split
        self._batch_size = batch_size
        self._num_workers = num_workers

        # split train test
        self._train_meta_df = self._meta_df[self._meta_df.train == True]
        self._test_meta_df = self._meta_df[self._meta_df.train == False]

        # split train val
        self._train_meta_df, self._val_meta_df = train_test_split(
            self._train_meta_df,
            train_size=self._train_split,
            random_state=random_state
        )

        self._train_video_paths = [os.path.join(self._prefix, row.database, row.file['path'], row.file['name']) for _, row in self._train_meta_df.iterrows()]
        self._val_video_paths = [os.path.join(self._prefix, row.database, row.file['path'], row.file['name']) for _, row in self._val_meta_df.iterrows()]
        self._test_video_paths = [os.path.join(self._prefix, row.database, row.file['path'], row.file['name']) for _, row in self._test_meta_df.iterrows()]

        # transforms for each video individually
        self._train_pre_transforms = [anyDictListToCompose(row.pre_transforms) for _, row in self._train_meta_df.iterrows()]
        self._val_pre_transforms = [anyDictListToCompose(row.pre_transforms) for _, row in self._val_meta_df.iterrows()]
        self._test_pre_transforms = [anyDictListToCompose(row.pre_transforms) for _, row in self._test_meta_df.iterrows()]

        self._train_aug_transforms = [anyDictListToCompose(row.aug_transforms) for _, row in self._train_meta_df.iterrows()]
        self._val_aug_transforms = [anyDictListToCompose(row.aug_transforms) for _, row in self._val_meta_df.iterrows()]
        self._test_aug_transforms = [anyDictListToCompose(row.aug_transforms) for _, row in self._test_meta_df.iterrows()]

        # store metadata
        self._train_metadata = train_metadata
        self._val_metadata = val_metadata
        self._test_metadata = test_metadata

    @property
    def metadata(self) -> Tuple[VideoDataset]:
        return (self._train_metadata, self._val_metadata, self._test_metadata)

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            self._train_set = VideoDataset(
                video_paths=self._train_video_paths,
                clip_length_in_frames=self._clip_length_in_frames,
                frames_between_clips=self._frames_between_clips,
                frame_rate=self._frame_rate,
                precomputed_metadata=self._train_metadata,
                num_workers=self._num_workers,
                pre_transforms=self._train_pre_transforms,
                aug_transforms=self._train_aug_transforms
            )

            self._train_set = self._train_set.metadata

            self._val_set = VideoDataset(
                video_paths=self._val_video_paths,
                clip_length_in_frames=self._clip_length_in_frames,
                frames_between_clips=self._frames_between_clips,
                frame_rate=self._frame_rate,
                precomputed_metadata=self._val_metadata,
                num_workers=self._num_workers,
                pre_transforms=self._val_pre_transforms,
                aug_transforms=self._val_aug_transforms,
                seeds=True
            )

            self._val_metadata = self._val_set.metadata

        if stage == 'test' or stage is None:
            self._test_set = VideoDataset(
                video_paths=self._test_video_paths,
                clip_length_in_frames=self._clip_length_in_frames,
                frames_between_clips=self._frames_between_clips,
                frame_rate=self._frame_rate,
                precomputed_metadata=self._test_metadata,
                num_workers=self._num_workers,
                pre_transforms=self._test_pre_transforms,
                aug_transforms=[None for _ in range(len(self._test_video_paths))],
                seeds=True
            )

            self._test_set = self._test_set.metadata

    def train_dataloader(self):
        return DataLoader(self._train_set, self._batch_size, shuffle=True, num_workers=self._num_workers, drop_last=True)  # shuffle train loader

    def val_dataloader(self):
        return DataLoader(self._val_set, self._batch_size, num_workers=self._num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self._test_set, self._batch_size, num_workers=self._num_workers, drop_last=True)

if __name__ == '__main__':
    import cv2
    import time
    from kornia import tensor_to_image
    
    from utils.io import save_pickle
 
    prefix = os.getcwd()
    paths = ['sample.mp4', 'sample.mp4', 'sample.mp4']
    
    meta_df = pd.DataFrame(columns=['database', 'train', 'file', 'pre_transforms', 'aug_transforms', 'auxiliary'])

    pre_transforms = [
        {'module': 'torchvision.transforms', 'type': 'Resize', 'kwargs': {'size': [270, 480]}}
        # {'module': 'kornia.augmentation', 'type': 'RandomGrayscale', 'kwargs': {'p': 0.5}}
    ]

    aug_transforms = [
        {'module': 'torchvision.transforms', 'type': 'RandomGrayscale', 'kwargs': {'p': 0.5}}
    ]

    for path in paths:
        row = {
            'database': '', # empty cause sample
            'train': True,
            'file': {'name': path, 'path': 'lightning_data_modules'},
            'pre_transforms': pre_transforms,
            'aug_transforms': aug_transforms,
            'auxiliary': {}
        }
        meta_df = meta_df.append(row, ignore_index=True)

    # create data module
    dm = VideoDataModule(
        meta_df=meta_df,
        prefix=prefix,
        clip_length_in_frames=20,
        frames_between_clips=1,
        frame_rate=8,
        batch_size=4
    )

    dm.setup('fit')
    metadata = dm.metadata
    # save_pickle('path.pkl', metadata)  # save and load metadata

    # get a sample dataloader
    dl = dm.train_dataloader()

    start = time.time_ns()

    for idx, batch in enumerate(dl):
        img, aug, fr, vid_fps, vid_idx, clip_idc = batch

        print('\rBatch {}/{}, img shape: {}, aug shape: {}, frame_rate: {}, video_fps: {}, Loading time: {}'.format(idx + 1, len(dl), img.shape, aug.shape, fr[0].item(), vid_fps[vid_idx[0]][0].item(), (time.time_ns() - start)/1.e9), end='')
        img = tensor_to_image(img[0, 0])
        aug = tensor_to_image(aug[0, 0])
        cv2.imshow('img', img)
        cv2.imshow('aug', aug)
        cv2.waitKey()
    cv2.destroyAllWindows()
