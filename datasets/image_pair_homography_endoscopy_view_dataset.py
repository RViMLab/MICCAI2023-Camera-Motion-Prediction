import os
import random
import imgaug
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from typing import List, Callable

from utils.transforms import RandomEdgeHomography, HOMOGRAPHY_RETURN, EndoscopyEllipsoid


class ImagePairHomographyEndoscopyViewDataset(Dataset):
    r"""Takes two images, supposedly from two time steps, and warps the second. Returns crops of both. 
    Implements the method described by DeTone et al. in https://arxiv.org/pdf/1606.03798.pdf.

    Args:
        df (pd.DataFrame): Pandas dataframe, must contain {'folder': , 'file': , 'vid': , 'frame': }
        prefix (str): Path to database e.g. </path/to/database>/df.folder/df.file
        rho (int): Image edges are randomly perturbed within [-rho, rho]
        crp_shape (list of int): Shape of cropped image
        p0 (float): Chance for homography being identity
        seq_len (int): Sequence length to sample images from, sequence length of 1 corresponds to static images, sequence length of 2 corresponds to neighboring images
        c_off_scale (List[float]): Center offset scale of image shape. Perturbes endoscopic view around image center
        min_scale (List[float]): Ellipsoid's half axes minimum scale
        max_scale (List[float]): Ellipsoid's half axes maximum scale
        min_rot (float): Ellipsoid's minimum rotation
        max_rot (float): Ellipsoid's maximum roation
        dc_scale (List[float]): Center update scale, center is perturbed by img.shape*dc_scale
        dori (List[float])
        update_chance (float): Chance by which ellipsoid's center and orientation are updated
        transforms (callable): Transforms to be applied before homography generation
        seeds (list of np.int32): Seeds for deterministic output, e.g. for test set
        return_img_pair (bool): Whether to return the original image pair

    Returns:
        if return_img_pair:
            dict (dict): (
                'img_pair' (torch.Tensor): Image pair of shape 2xCxHxW 
                'img_crp' (torch.Tensor): Crop of image img_pair[0] of shape Cx crp_shape[0] x crp_shape[1]
                'wrp_crp' (torch.Tensor): Crop of warp of image img_pair[1] of shape Cx crp_shape[0] x crp_shape[1]
                'uv' (torch.Tensor): Edges of crop of shape 4x2
                'duv' (torch.Tensor): Perturbation of edges uv within [-rho, rho] of shape 4x2
                'H' (torch.Tensor): Homography matrix of shape 3x3
            )
        else:
            dict (dict): (
                'img_crp' (torch.Tensor): Crop of image img_pair[0] of shape Cx crp_shape[0] x crp_shape[1]
                'wrp_crp' (torch.Tensor): Crop of warp of image img_pair[1] of shape Cx crp_shape[0] x crp_shape[1]
                'uv' (torch.Tensor): Edges of crop of shape 4x2
                'duv' (torch.Tensor): Perturbation of edges uv within [-rho, rho] of shape 4x2
                'H' (torch.Tensor): Homography matrix of shape 3x3
            )
    """
    def __init__(self, 
        df: pd.DataFrame, 
        prefix: str, 
        rho: int, 
        crp_shape: List[int], 
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
        transforms: Callable=None, 
        seeds: List[np.int32]=None, 
        return_img_pair: bool=True
    ) -> None:
        if seeds:
            if (len(df) != len(seeds)):
                raise Exception('In ImagePairHomographyEndoscopyViewDataset: Length of dataframe must equal length of seeds.')
        
        self._df = df.sort_values(['vid', 'frame']).reset_index(drop=True)
        self._prefix = prefix
        self._rho = rho
        self._reh = RandomEdgeHomography(rho=rho, crp_shape=crp_shape, p0=p0, homography_return=HOMOGRAPHY_RETURN.DATASET, seeds=seeds)
        if seq_len < 1:
            raise ValueError('Sequence length {} must be greater or equal 1.'.format(seq_len))
        self._seq_len = seq_len
        self._ee = EndoscopyEllipsoid()
        self._c_off_scale = c_off_scale
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._min_rot = min_rot
        self._max_rot = max_rot
        self._dc_scale = dc_scale
        self._dori = dori
        self._update_chance = update_chance
        self._transforms = transforms
        self._seeds = seeds
        self._return_image_pair = return_img_pair
        self._idcs = self._filterFeasibleSequenceIndices(self._df, col='vid', seq_len=self._seq_len)
        self._tt = ToTensor()

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, rho):
        self._rho = rho
        self._reh.rho = rho

    def __getitem__(self, idx):
        img_pair = []

        # set seed if desired
        if self._seeds:
            seed = self._seeds[idx]
        else:
            seed = random.randint(0, np.iinfo(np.int32).max)  # set random seed for numpy

        # randomly sample image pair
        np.random.seed(seed)
        idcs = self._idcs[idx] + np.random.choice(np.arange(self._seq_len), 2, replace=(self._seq_len == 1))  # static if self._seq_len = 1
        np.random.seed(None)

        file_pair = self._df.loc[idcs]

        for _, row in file_pair.iterrows():
            img = np.load(os.path.join(self._prefix, row.folder, row.file))
            
            if self._transforms:
                imgaug.seed(seed)
                img_pair.append(np.ascontiguousarray(self._transforms(img)))
            else:
                img_pair.append(img)

        # apply random edge homography
        self._reh.seed_idx = idx  # only uses seed index if self._reh._seeds is not None
        reh = self._reh(img_pair[1])

        img_crp = self._reh.crop(img_pair[0], reh['uv'])
        wrp_crp = reh['wrp_crp']

        img_crp, wrp_crp = self._ee.movingCenterPipeline(
            img=img_crp, wrp=wrp_crp,
            c_off_scale=self._c_off_scale,
            min_scale=self._min_scale, max_scale=self._max_scale,
            min_rot=self._min_rot, max_rot=self._max_rot,
            dc_scale=self._dc_scale, dori=self._dori, update_chance=self._update_chance,
            seed=seed
        )

        for i in range(len(img_pair)):
            img_pair[i] = self._tt(img_pair[i])

        img_crp = self._tt(img_crp)
        wrp_crp = self._tt(wrp_crp)

        if self._return_image_pair:
            return {
                'img_pair': img_pair,
                'img_crp': img_crp,
                'wrp_crp': wrp_crp,
                'uv': reh['uv'],
                'duv': reh['duv'],
                'H': reh['H']
            }
        else:
            return {
                'img_crp': img_crp,
                'wrp_crp': wrp_crp,
                'uv': reh['uv'], 
                'duv': reh['duv'], 
                'H': reh['H']
            }

    def __len__(self):
        return len(self._idcs)

    def _filterFeasibleSequenceIndices(self, 
        df: pd.DataFrame,
        col: str='vid',
        seq_len: int=2,
    ) -> pd.DataFrame:
        grouped_df = df.groupby(col)
        return grouped_df.apply(
            lambda x: x.iloc[seq_len-1:len(x) - (seq_len - 1)]  # get indices [seq_len - 1, length - (seq_len - 1)]
        ).index.get_level_values(1)  # return 2nd values of pd.MultiIndex
