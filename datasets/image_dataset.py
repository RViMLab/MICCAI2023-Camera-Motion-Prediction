import os
import random
from typing import Callable, List

import imgaug
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, prefix: str, transforms: List[Callable]=None, seeds: bool=False) -> None:
        self._df = df
        self._prefix = prefix
        self._transforms = transforms
        self._seeds = seeds

    def __getitem__(self, idx) -> T_co:
        # set seed if desired
        if self._seeds:
            seed = idx
        else:
            seed = random.randint(0, np.iinfo(np.int32).max) # set random seed for numpy
        row = self._df.iloc[idx]
        img = np.load(os.path.join(self._prefix, row.folder, row.file))
        if self._transforms:
            imgaug.seed(seed)
            img = self._transforms(img)
        return torch.from_numpy(img.transpose(2,0,1))  #HxWxC -> CxHxW

    def __len__(self):
        return len(self._df)
