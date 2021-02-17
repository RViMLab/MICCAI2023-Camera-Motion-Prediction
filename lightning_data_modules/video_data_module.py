import os
import pytorch_lightning as pl
from typing import List
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from datasets import VideoDataset
from utils.transforms import anyDictListToCompose


class VideoDataModule(pl.LightningDataModule):
    def __init__(self, meta_df: pd.DataFrame, prefix: str, clip_length_in_frames: int=25, frames_between_clips: int=1, train_split: float=0.8, batch_size: int=32, num_workers: int=2, random_state: int=42) -> None:
        r"""Pytorch Lightning datamodule for videos.
        
        Args:
            meta_df (pd.DataFrame): Meta dataframe containing {'database':, 'train':, 'file': {'name':, 'path':}, 'transforms': [], 'auxiliary':}
            prefix (str): Path to the database from meta_df
            clip_length_in_frames (int): Preview horizon, frames per returned clip
            frames_between_clips (int): Offset frames between starting point of clips
            train_slit (float): Relative size of train split
            batch_size (int): Batch size for the dataloader
            num_workers (int): Number of workers for the dataloader
            random_state (int): Initial random state for the train/validation split
        """
        self._meta_df = meta_df
        self._prefix = prefix

        self._clip_length_in_frames = clip_length_in_frames
        self._frames_between_clips = frames_between_clips

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
        self._train_transforms = [anyDictListToCompose(row.transforms) for _, row in self._train_meta_df.iterrows()]
        self._val_transforms = [anyDictListToCompose(row.transforms) for _, row in self._val_meta_df.iterrows()]
        self._test_transforms = [anyDictListToCompose(row.transforms) for _, row in self._test_meta_df.iterrows()]


    def setup(self, stage=None) -> None:
        if stage == 'fit' or stage is None:
            self._train_set = VideoDataset(
                video_paths=self._train_video_paths,
                clip_length_in_frames=self._clip_length_in_frames,
                frames_between_clips=self._frames_between_clips,
                transforms=self._train_transforms
            )

            self._val_set = VideoDataset(
                video_paths=self._val_video_paths,
                clip_length_in_frames=self._clip_length_in_frames,
                frames_between_clips=self._frames_between_clips,
                transforms=self._val_transforms,
                seeds=True
            )

        if stage == 'test' or stage is None:
            self._test_set = VideoDataset(
                video_paths=self._test_video_paths,
                clip_length_in_frames=self._clip_length_in_frames,
                frames_between_clips=self._frames_between_clips,
                transforms=self._test_transforms,
                seeds=True
            )

    def train_dataloader(self):
        return DataLoader(self._train_set, self._batch_size, num_workers=self._num_workers)

    def val_dataloader(self):
        return DataLoader(self._val_set, self._batch_size, num_workers=self._num_workers)

    def test_dataloader(self):
        return DataLoader(self._test_set, self._batch_size, num_workers=self._num_workers)

if __name__ == '__main__':
    import os
    import cv2
    from kornia import tensor_to_image
 
    prefix = os.getcwd()
    paths = ['sample.mp4', 'sample.mp4', 'sample.mp4']
    
    meta_df = pd.DataFrame(columns=['database', 'file', 'transforms'])
    transforms = [{'module': 'torchvision.transforms', 'type': 'Resize', 'kwargs': {'size': [270, 480]}}, {'module': 'kornia.augmentation', 'type': 'RandomGrayscale', 'kwargs': {'p': 0.5}}]

    for path in paths:
        row = {
            'database': '', # empty cause sample
            'train': True,
            'file': {'name': path, 'path': 'lightning_data_modules'},
            'transforms': transforms,
            'auxiliary': {}
        }
        meta_df = meta_df.append(row, ignore_index=True)

    # create data module
    dm = VideoDataModule(
        meta_df=meta_df,
        prefix=prefix,
        batch_size=4
    )

    dm.setup()

    # get a sample dataloader
    dl = dm.train_dataloader()

    for idx, batch in enumerate(dl):
        print('\rBatch {}/{}, Batch shape: {}'.format(idx + 1, len(dl),batch.shape), end='')
        img = batch[0,0]
        img = tensor_to_image(img)
        cv2.imshow('img', img)
        cv2.waitKey()
