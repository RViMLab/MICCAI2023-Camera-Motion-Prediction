import os
import pandas as pd
from torch.utils.data import Dataset
import imageio
from typing import Callable


class ImageSequenceDataset(Dataset):
    r"""Reads an images sequence from a image database.

    Args:
        df (pd.DataFrame): Pandas dataframe, must contain {'file_seq': [frame0.png, frame1.png], 'path': 'path/to/frames'}
        prefix (str): Path to database e.g. </path/to/database>/path/to/frames/frame.png
        transforms (callable): Transforms to be applied

    Returns:
        img_seq (list of Torch.tensor): Sequence of images of shape CxHxW
    """
    def __init__(self, df: pd.DataFrame, prefix: str, transforms: Callable=None):
        self.df = df
        self.prefix = prefix
        self.transforms = transforms

    def __getitem__(self, idx):
        file_seq = self.df['file_seq'][idx]
        img_seq = []

        for file in file_seq:
            img = imageio.imread(os.path.join(self.prefix, self.df['path'][idx], file))
            if self.transforms:
                img = self.transforms(img)
            img_seq.append(img)

        return img_seq

    def __len__(self):
        return len(self.df)
