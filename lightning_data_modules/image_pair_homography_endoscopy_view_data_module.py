import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
from typing import List

from datasets import ImagePairHomographyEndoscopyViewDataset
from utils.transforms import dictListToAugment


class ImagePairHomographyEndoscopyViewDataModule(pl.LightningDataModule):
    def __init__(self, 
            df: pd.DataFrame,
            prefix: str,
            train_split: float,
            batch_size: int,
            num_workers: int=2,
            rho: int=32,
            crp_shape: List[int]=[480, 640],
            p0: float=0.,
            seq_len: int=2,
            c_off_scale: List[float]=[0.125, 0.125], 
            min_scale: List[float]=[0.3, 0.3],
            max_scale: List[float]=[1.0, 1.0],
            min_rot: float=0.,
            max_rot: float=2*np.pi,
            dc_scale: List[float]=[0.1, 0.1],
            dori: List[float]=[-np.pi*0.1, np.pi*0.1],
            update_chance: float=1.0,
            unsupervised: bool=False,
            random_state: int=42,
            train_transforms: List[dict]=None,
            val_transforms: List[dict]=None,
            test_transforms: List[dict]=None,
            tolerance: float = 0.05
        ):
        super().__init__()
        # split into train and test set
        self._train_df = df[df['test'] == False]
        self._test_df = df[df['test'] == True].reset_index()

        # further split train into train and validation set
        unique_vid = self._train_df.vid.unique()

        train_vid, val_vid = train_test_split(
            unique_vid,
            train_size=train_split,
            random_state=random_state
        )

        self._val_df = self._train_df[self._train_df.vid.apply(lambda x: x in val_vid)].reset_index()
        self._train_df = self._train_df[self._train_df.vid.apply(lambda x: x in train_vid)].reset_index()

        # assert if fraction off
        fraction = len(self._val_df)/(len(self._train_df) + len(self._val_df))
        assert np.isclose(
            fraction, 1 - train_split, atol=tolerance
        ), 'Train set fraction {:.3f} not close enough to (1 - train_split) {} at tolerance {}'.format(fraction, 1 - train_split, tolerance)

        self._prefix = prefix
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._rho = rho
        self._crp_shape = crp_shape
        self._p0 = p0
        self._seq_len = seq_len
        self._c_off_scale = c_off_scale
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._min_rot = min_rot
        self._max_rot = max_rot
        self._dc_scale = dc_scale
        self._dori = dori
        self._update_chance = update_chance
        self._unsupervised = unsupervised

        self._train_transforms = dictListToAugment(train_transforms)
        self._val_transforms = dictListToAugment(val_transforms)
        self._test_transforms = dictListToAugment(test_transforms)

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, rho: int):
        self._rho = rho
        self._train_set.rho = rho
        self._val_set.rho = rho
        self._test_set.rho = rho

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self._train_set = ImagePairHomographyEndoscopyViewDataset(
                self._train_df, 
                self._prefix, 
                self._rho, 
                self._crp_shape,
                self._p0, 
                self._seq_len, 
                self._c_off_scale,
                self._min_scale,
                self._max_scale,
                self._min_rot,
                self._max_rot,
                self._dc_scale,
                self._dori,
                self._update_chance,
                transforms=self._train_transforms, 
                return_img_pair=self._unsupervised
            )
            seeds = np.arange(0, len(self._val_df)).tolist() # assure validation set is seeded the same for all epochs
            self._val_set = ImagePairHomographyEndoscopyViewDataset(
                self._val_df, 
                self._prefix, 
                self._rho, 
                self._crp_shape, 
                self._p0, 
                self._seq_len,
                self._c_off_scale,
                self._min_scale,
                self._max_scale,
                self._min_rot,
                self._max_rot,
                self._dc_scale,
                self._dori,
                self._update_chance,
                transforms=self._val_transforms, 
                seeds=seeds, 
                return_img_pair=True
            )
        if stage == 'test' or stage is None:
            seeds = np.arange(0, len(self._test_df)).tolist() # assure test set is seeded the same for all runs
            self._test_set = ImagePairHomographyEndoscopyViewDataset(
                self._test_df, 
                self._prefix, 
                self._rho, 
                self._crp_shape, 
                self._p0, 
                self._seq_len,
                self._c_off_scale,
                self._min_scale,
                self._max_scale,
                self._min_rot,
                self._max_rot,
                self._dc_scale,
                self._dori,
                self._update_chance,
                transforms=self._test_transforms, 
                seeds=seeds, 
                return_img_pair=self._unsupervised
            ) # for final evaluation

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch['img_crp'] = batch['img_crp'].to(device)
        batch['wrp_crp'] = batch['wrp_crp'].to(device)
        batch['duv'] = batch['duv'].to(device)
        if self._unsupervised:
            batch['img_pair'][0] = batch['img_pair'][0].to(device)
            batch['img_pair'][1] = batch['img_pair'][1].to(device)
            batch['uv'] = batch['uv'].to(device)
        return batch

    def train_dataloader(self):
        return DataLoader(self._train_set, batch_size=self._batch_size, num_workers=self._num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self._val_set, batch_size=self._batch_size, num_workers=self._num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self._test_set, batch_size=self._batch_size, num_workers=self._num_workers, pin_memory=True)


if __name__ == '__main__':
    import os
    import cv2
    from kornia import tensor_to_image
    from dotmap import DotMap
    
    from utils.io import load_yaml

    server = 'local'
    server = DotMap(load_yaml('config/servers.yml')[server])
    prefix = os.path.join(server.database.location, 'camera_motion_separated_npy/without_camera_motion')

    pkl_name = 'light_log_without_camera_motion.pkl'
    df = pd.read_pickle(os.path.join(prefix, pkl_name))

    cdm = ImagePairHomographyEndoscopyViewDataModule(df, prefix, train_split=0.8, batch_size=1, num_workers=0, rho=48, crp_shape=[240, 320])

    cdm.setup()
    train_dl = cdm.train_dataloader()

    for idx, batch in enumerate(train_dl):
        print('\ridx: {}, crp shape: {}, wrp shape: {}, len: {}'.format(idx, batch['img_crp'].shape, batch['wrp_crp'].shape, len(batch)), end='')
        cv2.imshow('img_crp', tensor_to_image(batch['img_crp'], False))
        cv2.imshow('wrp_crp', tensor_to_image(batch['wrp_crp'], False))
        cv2.waitKey()
