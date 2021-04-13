import os
import random
import imgaug
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from typing import List, Callable

from utils.transforms import RandomEdgeHomography, HOMOGRAPHY_RETURN, EndoscopyCircle


class ImagePairHomographyEndoscopyViewDataset(Dataset):
    r"""Takes two images, supposedly from two time steps, and warps the second. Returns crops of both. 
    Implements the method described by DeTone et al. in https://arxiv.org/pdf/1606.03798.pdf.

    Args:
        df (pd.DataFrame): Pandas dataframe, must contain {'file_seq': [frame0.png, frame1.png], 'path': 'path/to/frames'}
        prefix (str): Path to database e.g. </path/to/database>/path/to/frames/frame.png
        rho (int): Image edges are randomly perturbed within [-rho, rho]
        crp_shape (list of int): Shape of cropped image
        p0 (float): Chance for homography being identity
        c_off_scale (float): Center offset scale of image shape. Perturbes endoscopic view around image center
        dc_scale (float): Center update scale
        c_update_chance (float): Chance by which center is updated
        r_min_scale (float): Minimum radius, scale of minimum image size
        r_amp_scale (float): Radius amplitude, scale of maximum image size
        transforms (callable): Transforms to be applied before homography generation
        seeds (list of np.int32): Seeds for deterministic output, e.g. for test set

    Returns:
        dict (dict): (
            'img_pair' (torch.Tensor): Image pair of shape 2xCxHxW 
            'img_crp' (torch.Tensor): Crop of image img_pair[0] of shape Cx crp_shape[0] x crp_shape[1]
            'wrp_crp' (torch.Tensor): Crop of warp of image img_pair[1] of shape Cx crp_shape[0] x crp_shape[1]
            'uv' (torch.Tensor): Edges of crop of shape 4x2
            'duv' (torch.Tensor): Perturbation of edges uv within [-rho, rho] of shape 4x2
            'H' (torch.Tensor): Homography matrix of shape 3x3
        )
    """
    def __init__(self, df: pd.DataFrame, prefix: str, rho: int, crp_shape: List[int], p0: float=0., c_off_scale: float=0.125, dc_scale: float=0.1, c_update_chance: float=0.1, r_min_scale: float=0.25, r_amp_scale: float=0.5, transforms: Callable=None, seeds: List[np.int32]=None):
        if seeds:
            if (len(df) != len(seeds)):
                raise Exception('In ImagePairHomographyDataset: Length of dataframe must equal length of seeds.')
        
        self._df = df
        self._prefix = prefix   
        self._reh = RandomEdgeHomography(rho=rho, crp_shape=crp_shape, p0=p0, homography_return=HOMOGRAPHY_RETURN.DATASET, seeds=seeds)
        self._ec = EndoscopyCircle()
        self._c_off_scale = c_off_scale
        self._dc_scale = dc_scale
        self._c_update_chance = c_update_chance
        self._r_min_scale = r_min_scale
        self._r_amp_scale = r_amp_scale
        self._transforms = transforms
        self._seeds = seeds
        self._tt = ToTensor()

    def __getitem__(self, idx):
        file_seq = self._df['file_seq'][idx]
        img_pair = []

        if self._seeds:
            seed = self._seeds[idx]
        else:
            seed = random.randint(0, np.iinfo(np.int32).max)  # set random seed for numpy

        # randomly sample image pair
        np.random.seed(seed)
        file_pair = np.random.choice(file_seq, 2)
        np.random.seed(None)

        for file in file_pair:
            img = np.load(os.path.join(self._prefix, self._df['path'][idx], file))

            if self._transforms:
                imgaug.seed(seed)
                img_pair.append(np.ascontiguousarray(self._transforms(img)))
            else:
                img_pair.append(img)

        # apply random edge homography
        self._reh.seed_idx = idx
        reh = self._reh(img_pair[1])

        img_crp = self._reh.crop(img_pair[0], reh['uv'])
        wrp_crp = reh['wrp_crp']

        # apply endoscopy circle with moving center
        if self._seeds:  # test and validation
            seed = self._seeds[idx]
            img_crp, wrp_crp = self._ec.movingCenterPipeline(
                img=img_crp, wrp=wrp_crp, 
                c_off_scale=self._c_off_scale, dc_scale=self._dc_scale, c_update_chance=self._c_update_chance,
                r_min_scale=self._r_min_scale, r_amp_scale=self._r_amp_scale, 
                seed=seed
            )

        else:  # train
            seed = random.randint(0, np.iinfo(np.int32).max) # random seed via sampling
            img_crp, wrp_crp = self._ec.movingCenterPipeline(
                img=img_crp, wrp=wrp_crp, 
                c_off_scale=self._c_off_scale, dc_scale=self._dc_scale, c_update_chance=self._c_update_chance,
                r_min_scale=self._r_min_scale, r_amp_scale=self._r_amp_scale, 
                seed=seed
            )

        for i in range(len(img_pair)):
            img_pair[i] = self._tt(img_pair[i])

        img_crp = self._tt(img_crp)
        wrp_crp = self._tt(wrp_crp)

        return {
            'img_pair': img_pair,
            'img_crp': img_crp,
            'wrp_crp': wrp_crp,
            'uv': reh['uv'], 
            'duv': reh['duv'], 
            'H': reh['H']
        }

    def __len__(self):
        return len(self._df)
