import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from typing import List

from datasets import PandasHomographyDataset


class ConsecutiveDataModule(pl.LightningDataModule):
    def __init__(self, df: pd.DataFrame, prefix: str, train_split: float, batch_size: int, num_workers: int=2, rho: int=32, crp_shape: List[int]=[480, 640], unsupervised: bool=False):
        super().__init__()
        self.train_df = df[df['test'] == False].reset_index()
        self.test_df = df[df['test'] == True].reset_index()
        self.prefix = prefix
        self.train_split = train_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rho = rho
        self.crp_shape = crp_shape
        self.unsupervised = unsupervised

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            full_dataset = PandasHomographyDataset(self.train_df, self.prefix, self.rho, self.crp_shape)
            train_len = int(self.train_split*len(full_dataset))
            val_len = len(full_dataset) - train_len
            self.train_set, self.val_set = random_split(full_dataset, [train_len, val_len]) # for training and validation
        if stage == 'test' or stage is None:
            self.test_set = PandasHomographyDataset(self.test_df, self.prefix, self.rho, self.crp_shape) # for final evaluation

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

    cdm = ConsecutiveDataModule(df, prefix, train_split=0.8, batch_size=16, num_workers=0, rho=32, crp_shape=[640, 480])
    cdm.setup()

    for batch in cdm.train_dataloader():
        print(len(batch))
        print(batch['img_seq'][0].shape)
