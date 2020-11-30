import pandas as pd
import pytorch_lightning as pl
from typing import Callable
# from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from datasets import ImageSequenceDataset



# vid_to_png for single video
# fit lstm model to single video

# maybe add dataset that read from videos torch.io/opencv/nvidia dataloader/decord?
# find transforms for all cholec80 videos via processing.ipynb
# identify test sets -> update cholec80_transforms.yml with transforms and test flag
# convert whole cholec80 dataset into pngs ~2.6 TB
# use sequence_dataframe.ipynb to generate sequence dataframe, also include tool and phase annotation 
# train model on full dataset
# 
class SequenceDataModule(pl.LightningDataModule):
    def __init__(self, df: pd.DataFrame, prefix: str, train_split: float, batch_size: int, num_workers: int=2, train_trainsforms: Callable=None, val_transforms: Callable=None):
        self.df = df
        self.prefix = prefix
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transforms = train_trainsforms

    def setup(self, stage):
        if stage == 'fit':
            self.train_set = ImageSequenceDataset(df=self.df, prefix=self.prefix, transforms=self.train_transforms)
        if stage == 'test':
            pass

    def transfer_batch_to_device(self, batch, device):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
