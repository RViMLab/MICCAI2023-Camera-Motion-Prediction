import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
from typing import List

from datasets import ImagePairHomographyDataset
from utils.transforms import dictListToAugment


class ImagePairHomographyDataModule(pl.LightningDataModule):
    def __init__(self, df: pd.DataFrame, prefix: str, train_split: float, batch_size: int, num_workers: int=2, rho: int=32, crp_shape: List[int]=[480, 640], unsupervised: bool=False, random_state: int=42, train_transforms: List[dict]=None, val_transforms: List[dict]=None):
        super().__init__()
        self._train_df, self._val_df = train_test_split(
            df[df['test'] == False].reset_index(), 
            train_size=train_split, 
            random_state=random_state
        )
        self._train_df = self._train_df.reset_index()
        self._val_df = self._val_df.reset_index()
        self._test_df = df[df['test'] == True].reset_index()
        self._prefix = prefix
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._rho = rho
        self._crp_shape = crp_shape
        self._unsupervised = unsupervised

        self._train_transforms = dictListToAugment(train_transforms)
        self._val_transforms = dictListToAugment(val_transforms)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self._train_set = ImagePairHomographyDataset(self._train_df, self._prefix, self._rho, self._crp_shape, transforms=self._train_transforms)
            seeds = np.arange(0, len(self._val_df)).tolist() # assure validation set is seeded the same for all epochs
            self._val_set = ImagePairHomographyDataset(self._val_df, self._prefix, self._rho, self._crp_shape, transforms=self._val_transforms, seeds=seeds)
        if stage == 'test' or stage is None:
            seeds = np.arange(0, len(self._test_df)).tolist() # assure test set is seeded the same for all runs
            self._test_set = ImagePairHomographyDataset(self._test_df, self._prefix, self._rho, self._crp_shape, seeds=seeds) # for final evaluation

    def transfer_batch_to_device(self, batch, device):
        batch['img_crp'] = batch['img_crp'].to(device)
        batch['wrp_crp'] = batch['wrp_crp'].to(device)
        batch['duv'] = batch['duv'].to(device)
        if self._unsupervised:
            batch['img_pair'][0] = batch['img_pair'][0].to(device)
            batch['img_pair'][1] = batch['img_pair'][1].to(device)
            batch['uv'] = batch['uv'].to(device)
        return batch

    def train_dataloader(self):
        return DataLoader(self._train_set, batch_size=self._batch_size, num_workers=self._num_workers)

    def val_dataloader(self):
        return DataLoader(self._val_set, batch_size=self._batch_size, num_workers=self._num_workers)

    def test_dataloader(self):
        return DataLoader(self._test_set, batch_size=self._batch_size, num_workers=self._num_workers)


if __name__ == '__main__':
    import os

    prefix = '/media/martin/Samsung_T5/data/endoscopic_data/camera_motion_separated_png/without_camera_motion'
    pkl_name = 'log_without_camera_motion_seq_len_2.pkl'
    df = pd.read_pickle(os.path.join(prefix, pkl_name))

    cdm = ImagePairHomographyDataModule(df, prefix, train_split=0.8, batch_size=16, num_workers=0, rho=32, crp_shape=[640, 480])
    cdm.setup()

    for batch in cdm.train_dataloader():
        print(len(batch))
        print(batch['img_pair'][0].shape)
