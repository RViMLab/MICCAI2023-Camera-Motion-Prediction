from typing import List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from datasets import ImageSequenceDataset, ImageSequenceDuvDataset
from utils.transforms import dictListToAugment


class ImageSequenceDataModule(pl.LightningDataModule):
    def __init__(self,
        df: pd.DataFrame, 
        prefix: str,
        train_split: float, 
        batch_size: int, 
        num_workers: int=2,
        random_state: int=42,
        tolerance: float = 0.05,
        seq_len: int=10,
        frame_increment: int=1,
        frames_between_clips: int=1,
        random_frame_offset: bool=False,
        train_transforms: List[dict]=None, 
        val_transforms: List[dict]=None, 
        test_transforms: List[dict]=None,
        load_images: bool=True
    ):
        super().__init__()

        # train/test
        self._train_df = df[df.train == True]
        self._test_df = df[df.train == False]

        # further split train into train and validation set
        unique_vid = self._train_df.vid.unique()

        train_vid, val_vid = train_test_split(
            unique_vid,
            train_size=train_split,
            random_state=random_state
        )

        self._val_df = self._train_df[self._train_df.vid.apply(lambda x: x in val_vid)].reset_index()
        self._train_df = self._train_df[self._train_df.vid.apply(lambda x: x in train_vid)].reset_index()

        # assert if fraction off
        fraction = len(self._val_df)/(len(self._train_df) + len(self._val_df))
        assert np.isclose(
            fraction, 1 - train_split, atol=tolerance
        ), 'Train set fraction {:.3f} not close enough to (1 - train_split) {} at tolerance {}'.format(fraction, 1 - train_split, tolerance)

        self._prefix = prefix
        self._batch_size = batch_size
        self._num_workers = num_workers

        self._seq_len = seq_len
        self._frame_increment = frame_increment
        self._frames_between_clips = frames_between_clips
        self._random_frame_offset = random_frame_offset

        self._train_tranforms = dictListToAugment(train_transforms)
        self._val_transforms = dictListToAugment(val_transforms)
        self._test_transforms = dictListToAugment(test_transforms)

        self._load_images = load_images

    def setup(self, stage: str=None) -> None:
        if stage == 'fit' or stage is None:
            self._train_set = ImageSequenceDataset(
                df=self._train_df,
                prefix=self._prefix,
                seq_len=self._seq_len,
                frame_increment=self._frame_increment,
                frames_between_clips=self._frames_between_clips,
                random_frame_offset=self._random_frame_offset,
                transforms=self._train_tranforms,
                load_images=self._load_images,
                seeds=False
            )
            self._val_set = ImageSequenceDataset(
                df=self._val_df,
                prefix=self._prefix,
                seq_len=self._seq_len,
                frame_increment=self._frame_increment,
                frames_between_clips=self._frames_between_clips,
                random_frame_offset=False,
                transforms=self._val_transforms,
                load_images=self._load_images,
                seeds=True
            )
        if stage == 'test':
            self._test_set = ImageSequenceDataset(
                df=self._test_df,
                prefix=self._prefix,
                seq_len=self._seq_len,
                frame_increment=self._frame_increment,
                frames_between_clips=self._frames_between_clips,
                random_frame_offset=False,
                transforms=self._test_transforms, 
                load_images=self._load_images,
                seeds=True
            )

    # def transfer_batch_to_device(self, batch, device, dataloader_idx):
    #     pass

    def train_dataloader(self):
        return DataLoader(self._train_set, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self._val_set, batch_size=self._batch_size, num_workers=self._num_workers, drop_last=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self._test_set, batch_size=self._batch_size, num_workers=self._num_workers, drop_last=True, pin_memory=True)


class ImageSequenceDuvDataModule(pl.LightningDataModule):
    def __init__(self,
        df: pd.DataFrame, 
        prefix: str,
        train_split: float, 
        batch_size: int, 
        num_workers: int=2,
        random_state: int=42,
        tolerance: float = 0.05,
        seq_len: int=10,
        frame_increment: int=1,
        frames_between_clips: int=1,
        random_frame_offset: bool=False,
        train_transforms: List[dict]=None, 
        val_transforms: List[dict]=None, 
        test_transforms: List[dict]=None,
        load_images: bool=True
    ):
        super().__init__()

        # train/test
        self._train_df = df[df.train == True]
        self._test_df = df[df.train == False]

        # further split train into train and validation set
        unique_vid = self._train_df.vid.unique()

        train_vid, val_vid = train_test_split(
            unique_vid,
            train_size=train_split,
            random_state=random_state
        )

        self._val_df = self._train_df[self._train_df.vid.apply(lambda x: x in val_vid)].reset_index()
        self._train_df = self._train_df[self._train_df.vid.apply(lambda x: x in train_vid)].reset_index()

        # assert if fraction off
        fraction = len(self._val_df)/(len(self._train_df) + len(self._val_df))
        assert np.isclose(
            fraction, 1 - train_split, atol=tolerance
        ), 'Train set fraction {:.3f} not close enough to (1 - train_split) {} at tolerance {}'.format(fraction, 1 - train_split, tolerance)

        self._prefix = prefix
        self._batch_size = batch_size
        self._num_workers = num_workers

        self._seq_len = seq_len
        self._frame_increment = frame_increment
        self._frames_between_clips = frames_between_clips
        self._random_frame_offset = random_frame_offset

        self._train_tranforms = dictListToAugment(train_transforms)
        self._val_transforms = dictListToAugment(val_transforms)
        self._test_transforms = dictListToAugment(test_transforms)

        self._load_images = load_images

    def setup(self, stage: str=None) -> None:
        if stage == 'fit':
            self._train_set = ImageSequenceDuvDataset(
                df=self._train_df,
                prefix=self._prefix,
                seq_len=self._seq_len,
                frame_increment=self._frame_increment,
                frames_between_clips=self._frames_between_clips,
                random_frame_offset=self._random_frame_offset,
                transforms=self._train_tranforms, 
                load_images=self._load_images,
                seeds=False
            )
            self._val_set = ImageSequenceDuvDataset(
                df=self._val_df,
                prefix=self._prefix,
                seq_len=self._seq_len,
                frame_increment=self._frame_increment,
                frames_between_clips=self._frames_between_clips,
                random_frame_offset=False,
                transforms=self._val_transforms,
                load_images=True,
                seeds=True
            )
        if stage == 'test':
            self._test_set = ImageSequenceDuvDataset(
                df=self._test_df,
                prefix=self._prefix,
                seq_len=self._seq_len,
                frame_increment=self._frame_increment,
                frames_between_clips=self._frames_between_clips,
                random_frame_offset=False,
                transforms=self._test_transforms,
                load_images=True,
                seeds=True
            )

    def train_dataloader(self):
        return DataLoader(self._train_set, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self._val_set, batch_size=self._batch_size, num_workers=self._num_workers, drop_last=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self._test_set, batch_size=self._batch_size, num_workers=self._num_workers, drop_last=True, pin_memory=True)


if __name__ == "__main__":
    import cv2
    from kornia import tensor_to_image

    from utils.processing import unique_video_train_test
        
    df = pd.read_pickle("/media/martin/Samsung_T5/data/endoscopic_data/cholec80_frames/log.pkl")
    df = unique_video_train_test(df)
    prefix = "/media/martin/Samsung_T5/data/endoscopic_data/cholec80_frames"

    dm = ImageSequenceDataModule(
        df, prefix, train_split=0.8, batch_size=10, tolerance=0.2, seq_len=10
    )
    dm.setup()

    for frames, frames_transformed, idcs, vid_idx in dm.train_dataloader():
        print(idcs)
        for frame in frames[0]:
            frame = tensor_to_image(frame, False)
            cv2.imshow('frame', frame)
            cv2.waitKey()
