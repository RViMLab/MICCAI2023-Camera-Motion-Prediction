from typing import List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset, random_split

from datasets import ImagePairHomographyDataset
from utils.transforms import dict_list_to_augment


class ImagePairHomographyDataModule(pl.LightningDataModule):
    def __init__(self, 
            df: pd.DataFrame,
            prefix: str,
            train_split: float,
            batch_size: int,
            num_workers: int=2,
            rho: int=32,
            crp_shape: List[int]=[480, 640],
            p0: float=0.,
            seq_len: int=2,
            unsupervised: bool=False,
            random_state: int=42,
            train_transforms: List[dict]=None,
            val_transforms: List[dict]=None,
            test_transforms: List[dict]=None,
            tolerance: float = 0.05
        ):
        super().__init__()
        # split into train and test set
        self._train_df = df[df['test'] == False]
        self._test_df = df[df['test'] == True].reset_index()

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
        self._rho = rho
        self._crp_shape = crp_shape
        self._p0 = p0
        self._seq_len = seq_len
        self._unsupervised = unsupervised

        self._train_transforms = dict_list_to_augment(train_transforms)
        self._val_transforms = dict_list_to_augment(val_transforms)
        self._test_transforms = dict_list_to_augment(test_transforms)

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, rho: int):
        self._rho = rho
        self._train_set.rho = rho
        self._val_set.rho = rho
        self._test_set.rho = rho

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self._train_set = ImagePairHomographyDataset(self._train_df, self._prefix, self._rho, self._crp_shape, self._p0, self._seq_len, transforms=self._train_transforms, return_img_pair=self._unsupervised)
            seeds = np.arange(0, len(self._val_df)).tolist() # assure validation set is seeded the same for all epochs
            self._val_set = ImagePairHomographyDataset(self._val_df, self._prefix, self._rho, self._crp_shape, self._p0, self._seq_len, transforms=self._val_transforms, seeds=seeds, return_img_pair=True)
        if stage == 'test' or stage is None:
            seeds = np.arange(0, len(self._test_df)).tolist() # assure test set is seeded the same for all runs
            self._test_set = ImagePairHomographyDataset(self._test_df, self._prefix, self._rho, self._crp_shape, self._p0, self._seq_len, transforms=self._test_transforms, seeds=seeds, return_img_pair=self._unsupervised) # for final evaluation

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch['img_crp'] = batch['img_crp'].to(device)
        batch['wrp_crp'] = batch['wrp_crp'].to(device)
        batch['duv'] = batch['duv'].to(device)
        if self._unsupervised:
            batch['img_pair'][0] = batch['img_pair'][0].to(device)
            batch['img_pair'][1] = batch['img_pair'][1].to(device)
            batch['uv'] = batch['uv'].to(device)
        return batch

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self._train_set, batch_size=self._batch_size, num_workers=self._num_workers, pin_memory=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self._val_set, batch_size=self._batch_size, num_workers=self._num_workers, pin_memory=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self._test_set, batch_size=self._batch_size, num_workers=self._num_workers, pin_memory=True)


if __name__ == '__main__':
    import os
    import time

    from dotmap import DotMap

    from utils.io import load_yaml

    server = 'local'
    server = DotMap(load_yaml('config/servers.yml')[server])
    prefix = os.path.join(server.database.location, 'camera_motion_separated_npy/without_camera_motion')

    pkl_name = 'light_log_without_camera_motion.pkl'
    # pkl_name = 'light_log_without_camera_motion.pkl'
    df = pd.read_pickle(os.path.join(prefix, pkl_name))

    cdm = ImagePairHomographyDataModule(df, prefix, train_split=0.8, batch_size=64, num_workers=0, rho=32, crp_shape=[320, 240])
    # cdm = ImagePairHomographyDataModuleSequenceDf(df, prefix, train_split=0.8, batch_size=64, num_workers=0, rho=32, crp_shape=[320, 240])
    cdm.setup()
    train_dl = cdm.train_dataloader()

    start = time.time_ns()
    for idx, batch in enumerate(train_dl):
        print('\ridx: {}, crp shape: {}, wrp shape: {}, len: {}'.format(idx, batch['img_crp'].shape, batch['wrp_crp'].shape, len(batch)), end='')
        if idx == 9:
            break
    print('\nTime taken: {}'.format((time.time_ns() - start)/1.e9))
