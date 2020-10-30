import os
import cv2
import pandas as pd
from torch.utils.data import Dataset
from typing import List

from utils.transforms import RandomEdgeHomography, HOMOGRAPHY_RETURN


class PandasHomographyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, prefix: str, rho: int, crp_shape: List[int] , transforms=None):
        self.df = df
        self.prefix = prefix
        self.reh = RandomEdgeHomography(rho=rho, crp_shape=crp_shape, homography_return=HOMOGRAPHY_RETURN.DATASET)
        self.transforms = transforms

    def __getitem__(self, idx):
        file_seq = self.df['file_seq'][idx]
        img_seq = []

        for file in file_seq:
            img_seq.append(cv2.imread(os.path.join(self.prefix, self.df['path'][idx], file)))

        if self.transforms:
            for i in range(len(img_seq)):
                img_seq[i] = self.transforms(img_seq[i])

        # apply random edge homography
        reh = self.reh(img_seq[1])

        img_seq[0] = self.reh.crop(img_seq[0], reh['uv'])
        img_seq[1] = reh['wrp_crp']

        return {
            'img_seq': img_seq, 
            'uv': reh['uv'], 
            'duv': reh['duv'], 
            'H': reh['H']
        }

    def __len__(self):
        return len(self.df)
