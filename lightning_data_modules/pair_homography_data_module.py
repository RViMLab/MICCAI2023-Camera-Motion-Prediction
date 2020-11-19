import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
from typing import List

from datasets import PairHomographyDataset
from utils.transforms import dict_list_to_augment_image


class PairHomographyDataModule(pl.LightningDataModule):
    def __init__(self, df: pd.DataFrame, prefix: str, train_split: float, batch_size: int, num_workers: int=2, rho: int=32, crp_shape: List[int]=[480, 640], unsupervised: bool=False, random_state: int=42, train_transforms=None, val_transforms=None):
        super().__init__()
        self.train_df, self.val_df = train_test_split(
            df[df['test'] == False].reset_index(), 
            train_size=train_split, 
            random_state=random_state
        )
        self.train_df = self.train_df.reset_index()
        self.val_df = self.val_df.reset_index()
        self.test_df = df[df['test'] == True].reset_index()
        self.prefix = prefix
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rho = rho
        self.crp_shape = crp_shape
        self.unsupervised = unsupervised

        self.train_transforms = dict_list_to_augment_image(train_transforms)
        self.val_transforms = dict_list_to_augment_image(val_transforms)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set = PairHomographyDataset(self.train_df, self.prefix, self.rho, self.crp_shape, transforms=self.train_transforms)
            seeds = np.arange(0, len(self.val_df)) # assure validation set is seeded the same for all epochs
            self.val_set = PairHomographyDataset(self.val_df, self.prefix, self.rho, self.crp_shape, transforms=self.val_transforms, seeds=seeds)
        if stage == 'test' or stage is None:
            seeds = np.arange(0, len(self.test_df)) # assure test set is seeded the same for all runs
            self.test_set = PairHomographyDataset(self.test_df, self.prefix, self.rho, self.crp_shape, seeds=seeds) # for final evaluation

    def transfer_batch_to_device(self, batch, device):
        batch['img_seq_crp'][0] = batch['img_seq_crp'][0].to(device)
        batch['img_seq_crp'][1] = batch['img_seq_crp'][1].to(device)
        batch['duv'] = batch['duv'].to(device)
        if self.unsupervised:
            batch['img_seq'][0] = batch['img_seq'][0].to(device)
            batch['img_seq'][1] = batch['img_seq'][1].to(device)
            batch['uv'] = batch['uv'].to(device)
        return batch

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == '__main__':
    import os

    prefix = '/media/martin/Samsung_T5/data/endoscopic_data/camera_motion_separated_png/without_camera_motion'
    pkl_name = 'log_without_camera_motion_seq_len_2.pkl'
    df = pd.read_pickle(os.path.join(prefix, pkl_name))

    cdm = PairHomographyDataModule(df, prefix, train_split=0.8, batch_size=16, num_workers=0, rho=32, crp_shape=[640, 480])
    cdm.setup()

    for batch in cdm.train_dataloader():
        print(len(batch))
        print(batch['img_seq'][0].shape)
