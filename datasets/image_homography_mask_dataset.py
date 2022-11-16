import os
import random
from typing import Callable, List

import imgaug
import kornia
import numpy as np
import pandas as pd
import torch
from pytorch_lightning.utilities.seed import reset_seed, seed_everything
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from utils import four_point_homography_to_matrix, image_edges


class ImageHomographyMaskDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame,
        prefix: str,
        rho: int,
        transforms: List[Callable]=None,
        seeds: bool=False
    ) -> None:
        self._df = df
        self._prefix = prefix
        self._rho = rho
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

        img = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0) # HxWxC -> 1xCxHxW
        img = img.to(torch.float32)/255.

        mask = torch.ones((1, 1,) + img.shape[-2:], dtype=img.dtype)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        duv = torch.randint(-self._rho, self._rho, [1,4,2])
        try:
            H = four_point_homography_to_matrix(image_edges(img), duv)
            mask = kornia.geometry.warp_perspective(mask, H, mask.shape[-2:])
            return img.squeeze(0), mask.squeeze(0)
        except:
            return img.squeeze(0), mask.squeeze(0)
            

    def __len__(self):
        return len(self._df)
