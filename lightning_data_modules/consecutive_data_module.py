import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets import PandasDataset

class ConsecutiveDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, num_workers: int):
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, , state=None):
        self.train_set = PandasDataset(self.df_path, )
        self.val_set = # for evaluation midst training
        self.test_set = # for final evaluation

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)
