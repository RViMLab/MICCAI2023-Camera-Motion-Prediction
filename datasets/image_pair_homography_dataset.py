import os
import imageio
import imgaug
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from typing import List, Callable

from utils.transforms import RandomEdgeHomography, HOMOGRAPHY_RETURN, EndoscopyCircle


class ImagePairHomographyDataset(Dataset):
    r"""Takes two images, supposedly from two time steps, and warps the second. Returns crops of both. 
    Implements the method described by DeTone et al. in https://arxiv.org/pdf/1606.03798.pdf.

    Args:
        df (pd.DataFrame): Pandas dataframe, must contain {'file_seq': [frame0.png, frame1.png], 'path': 'path/to/frames'}
        prefix (str): Path to database e.g. </path/to/database>/path/to/frames/frame.png
        rho (int): Image edges are randomly perturbed within [-rho, rho]
        crp_shape (list of int): Shape of cropped image
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
    def __init__(self, df: pd.DataFrame, prefix: str, rho: int, crp_shape: List[int], c_off_scale: float=0.125, dc_scale: float=0.1, c_update_chance: float=0.1, r_min_scale: float=0.25, r_max: float=0.5, transforms: Callable=None, seeds: List[np.int32]=None):
        if seeds:
            if (len(df) != len(seeds)):
                raise Exception('In ImagePairHomographyDataset: Length of dataframe must equal length of seeds.')

        if (len(df['file_seq'][0]) != 2):
            raise Exception('In ImagePairHomographyDataset: Length of file_seq in dataframe must equal 2.')
        
        self._df = df
        self._prefix = prefix   
        self._reh = RandomEdgeHomography(rho=rho, crp_shape=crp_shape, homography_return=HOMOGRAPHY_RETURN.DATASET, seeds=seeds)
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

        for file in file_seq:
            img = imageio.imread(os.path.join(self._prefix, self._df['path'][idx], file))
            img_pair.append(img)

        if self._transforms:
            seed = np.random.randint(np.iinfo(np.int32).max) # set random seed for numpy
            for i in range(len(img_pair)):
                imgaug.seed(seed)
                img_pair[i] = np.ascontiguousarray(self._transforms(img_pair[i]))

        # apply random edge homography
        self._reh.seed_idx = idx
        reh = self._reh(img_pair[1])

        img_crp = self._reh.crop(img_pair[0], reh['uv'])
        wrp_crp = reh['wrp_crp']

        # apply endoscopy circle with moving center
        if self._seeds:  # test and validation
            seed = self._seeds[idx]
            center = EndoscopyCircle.randomCenter(shape=img_crp.shape, c_off_scale=self._c_off_scale, seed=seed)
            radius = EndoscopyCircle.randomRadius(shape=img_crp.shape, r_min_scale=self._r_min_scale, r_amp_scale=self._r_amp_scale, seed=seed)  # assure same radius

            img_crp = self._ec(img_crp, center=center, radius=radius)

            # update center by chance
            center = EndoscopyCircle.randomCenterUpdate(img_crp.shape, center=center, scale=self._dc_scale, chance=self._c_update_chance, seed=seed)
            wrp_crp = self._ec(wrp_crp, center=center, radius=radius)
        else:  # train
            seed = np.random.randint(np.iinfo(np.int32).max) # random seed via sampling
            center = EndoscopyCircle.randomCenter(shape=img_crp.shape, c_off_scale=self._c_off_scale, seed=seed)
            radius = EndoscopyCircle.randomRadius(shape=img_crp.shape, r_min_scale=self._r_min_scale, r_amp_scale=self._r_amp_scale, seed=seed)  # assure same radius

            img_crp = self._ec(img_crp, center=center, seed=seed)

            # update center by chance
            center = EndoscopyCircle.randomCenterUpdate(img_crp.shape, center=center, scale=self._dc_scale, chance=self._c_update_chance, seed=seed)
            wrp_crp = self._ec(wrp_crp, center=center, seed=seed)

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
