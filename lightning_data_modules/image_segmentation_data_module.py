import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Callable

from datasets import ImageSegmentationDataset
from utils.transforms import dictListToAugment


class ImageSegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        df: pd.DataFrame,
        prefix: str,
        train_split: float=0.8,
        batch_size: int=16,
        num_workers: int=2,
        random_state: int=42,
        train_image_transforms: Callable=None, 
        train_spatial_transforms: Callable=None, 
        val_image_transforms: Callable=None,
        val_spatial_transforms: Callable=None,
        test_image_transforms: Callable=None,
        test_spatial_transforms: Callable=None
    ):
        super().__init__()

        self._prefix = prefix
        self._train_split = train_split
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._random_state = random_state
        self._train_image_transforms = dictListToAugment(train_image_transforms)
        self._train_spatial_transforms = dictListToAugment(train_spatial_transforms)
        self._val_image_transforms = dictListToAugment(val_image_transforms)
        self._val_spatial_transforms = dictListToAugment(val_spatial_transforms)
        self._test_image_transforms = dictListToAugment(test_image_transforms)
        self._test_spatial_transforms = dictListToAugment(test_spatial_transforms)

        # split train test
        self._train_df = df[df.test == False]
        self._test_df = df[df.test == True].reset_index()

        # split train val
        self._train_df, self._val_df = train_test_split(
            self._train_df,
            train_size=self._train_split,
            random_state=self._random_state
        )

        self._train_df, self._val_df = self._train_df.reset_index(), self._val_df.reset_index()

    def setup(self, stage: str=None):
        if stage == 'fit' or stage is None:
            self._train_set = ImageSegmentationDataset(self._train_df, self._prefix, image_transforms=self._train_image_transforms, spatial_transforms=self._train_spatial_transforms)
            seeds = np.arange(0, len(self._val_df)).tolist() # assure validation set is seeded the same for all epochs
            self._val_set = ImageSegmentationDataset(self._val_df, self._prefix, image_transforms=self._val_image_transforms, spatial_transforms=self._val_spatial_transforms, seeds=seeds)
        if stage == 'test' or stage is None:
            seeds = np.arange(0, len(self._test_df)).tolist() # assure test set is seeded the same for all runs
            self._test_set = ImageSegmentationDataset(self._test_df, self._prefix, image_transforms=self._test_image_transforms, spatial_transforms=self._test_spatial_transforms, seeds=seeds)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
        return batch

    def train_dataloader(self):
        return DataLoader(self._train_set, batch_size=self._batch_size, num_workers=self._num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self._val_set, batch_size=self._batch_size, num_workers=self._num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self._test_set, batch_size=self._batch_size, num_workers=self._num_workers, pin_memory=True)


if __name__ == '__main__':
    import os
    import cv2
    from kornia import tensor_to_image

    from utils.io import load_yaml

    prefix = '/media/martin/Samsung_T5/data/endoscopic_data/boundary_segmentation'

    df = pd.read_pickle(os.path.join(prefix, 'light_log.pkl'))

    config = load_yaml('config/boundary_segmentation.yml')

    image_transforms = config['data']['image_transforms']
    spatial_transforms = config['data']['spatial_transforms']

    dm = ImageSegmentationDataModule(
        df,
        prefix,
        batch_size=1,
        train_image_transforms=image_transforms,
        train_spatial_transforms=spatial_transforms,
        val_image_transforms=image_transforms,
        val_spatial_transforms=spatial_transforms
    )

    dm.setup()

    dl = dm.train_dataloader()

    for batch in dl:
        img, seg = batch

        img, seg = tensor_to_image(img, keepdim=False), tensor_to_image(seg, keepdim=False)
        cv2.imshow('img', img)
        cv2.imshow('seg', seg)
        cv2.waitKey()
