import pytorch_lightning as pl
from typing import List
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from datasets import VideoDataset
from utils.transforms import anyDictListToCompose


class VideoDataModule(pl.LightningDataModule):
    def __init__(self, train_video_paths: List[str], test_video_paths: List[str], clip_length_in_frames: int=25, frames_between_clips: int=1, train_split: float=0.8, batch_size: int=32, num_workers: int=2, random_state: int=42, video_specific_train_transforms: List[List[dict]]=None, video_specific_val_transforms: List[List[dict]]=None) -> None:
        r"""Pytorch Lightning datamodule for image pairs in video sequences.
        
        Args:
            train_video_paths (List[str]): List of paths to train video files
            test_video_paths (List[str]): Same as train_video_paths but for testing
            clip_length_in_frames (int): Preview horizon, frames per returned clip
            frames_between_clips (int): Offset frames between starting point of clips
            train_slit (float): Relative size of train split
            batch_size (int): Batch size for the dataloader
            num_workers (int): Number of workers for the dataloader
            random_state (int): Initial random state for the train/validation split
            video_specifictrain_transforms (List[List[dict]]): List of list of dictionaries for training, e.g. [[{'module': 'some.module', 'type': 'callable', 'kwargs': {'key': val}}]
            video_specificval_transforms (List[List[dict]]): List of list of dictionaries for validation, e.g. [[{'module': 'some.module', 'type': 'callable', 'kwargs': {'key': val}}]
        """

        self._train_video_paths, self._val_video_paths = train_test_split(
            train_video_paths, 
            train_size=train_split,
            random_state=random_state
        )
        self._test_video_paths = test_video_paths

        self._clip_length_in_frames = clip_length_in_frames
        self._frames_between_clips = frames_between_clips

        self._batch_size = batch_size
        self._num_workers = num_workers

        # transforms for each video individually
        self._train_transforms = []
        self._val_transforms = []

        if video_specific_train_transforms is not None:
            for transforms in video_specific_train_transforms:
                self._train_transforms.append(anyDictListToCompose(transforms))
        else:
            self._train_transforms = None
        if video_specific_val_transforms is not None:
            for transforms in video_specific_val_transforms:
                self._val_transforms.append(anyDictListToCompose(transforms))
        else:
            self._val_transforms = None

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
    paths = [os.path.join(prefix, 'lightning_data_modules', x) for x in paths]

    # create data module
    dm = VideoImagePairsDataModule(
        paths[:2], paths[2:],
        batch_size=4,
        video_specific_train_transforms=[
            [{'module': 'torchvision.transforms', 'type': 'Resize', 'kwargs': {'size': [270, 480]}}, {'module': 'kornia.augmentation', 'type': 'RandomGrayscale', 'kwargs': {'p': 0.5}}],
        ]
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
