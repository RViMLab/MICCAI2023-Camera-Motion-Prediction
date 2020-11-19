import os
import cv2
import imgaug
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from typing import List

from utils.transforms import RandomEdgeHomography, HOMOGRAPHY_RETURN


class PairHomographyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, prefix: str, rho: int, crp_shape: List[int] , transforms=None, seeds: List[np.int32]=None):
        self.df = df
        self.prefix = prefix
        self.reh = RandomEdgeHomography(rho=rho, crp_shape=crp_shape, homography_return=HOMOGRAPHY_RETURN.DATASET, seeds=seeds)
        self.transforms = transforms
        self.tt = ToTensor()

    def __getitem__(self, idx):
        file_seq = self.df['file_seq'][idx]
        img_seq = []
        img_seq_crp = []

        for file in file_seq:
            img = cv2.imread(os.path.join(self.prefix, self.df['path'][idx], file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_seq.append(img)

        if self.transforms:
            seed = np.random.randint(np.iinfo(np.int32).max) # set random seed for numpy
            for i in range(len(img_seq)):
                imgaug.seed(seed)
                img_seq[i] = np.ascontiguousarray(self.transforms(img_seq[i]))

        # apply random edge homography
        reh = self.reh(img_seq[1], idx)

        img_seq_crp.append(self.reh.crop(img_seq[0], reh['uv']))
        img_seq_crp.append(reh['wrp_crp'])

        for i in range(len(img_seq)):
            img_seq[i] = self.tt(img_seq[i])
            img_seq_crp[i] = self.tt(img_seq_crp[i])

        return {
            'img_seq': img_seq,
            'img_seq_crp': img_seq_crp, 
            'uv': reh['uv'], 
            'duv': reh['duv'], 
            'H': reh['H']
        }

    def __len__(self):
        return len(self.df)