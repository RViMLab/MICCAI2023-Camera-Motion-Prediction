import os
import imageio
import imgaug
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from typing import List, Callable

from utils.transforms import RandomEdgeHomography, HOMOGRAPHY_RETURN


class ImagePairHomographyDataset(Dataset):
    r"""Takes two images, supposedly from two time steps, and warps the second. Returns crops of both. 
    Implements the method described by DeTone et al. in https://arxiv.org/pdf/1606.03798.pdf.

    Args:
        df (pd.DataFrame): Pandas dataframe, must contain {'file_seq': [frame0.png, frame1.png], 'path': 'path/to/frames'}
        prefix (str): Path to database e.g. </path/to/database>/path/to/frames/frame.png
        rho (int): Image edges are randomly perturbed within [-rho, rho]
        crp_shape (list of int): Shape of cropped image
        transforms (callable): Transforms to be applied before homography generation
        seeds (list of np.int32): Seeds for deterministic output

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
    def __init__(self, df: pd.DataFrame, prefix: str, rho: int, crp_shape: List[int] , transforms: Callable=None, seeds: List[np.int32]=None):
        if seeds:
            if (len(df) != len(seeds)):
                raise Exception('In ImagePairHomographyDataset: Length of dataframe must equal length of seeds.')

        if (len(df['file_seq'][0]) != 2):
            raise Exception('In ImagePairHomographyDataset: Length of file_seq in dataframe must equal 2.')
        
        self.df = df
        self.prefix = prefix   
        self.reh = RandomEdgeHomography(rho=rho, crp_shape=crp_shape, homography_return=HOMOGRAPHY_RETURN.DATASET, seeds=seeds)
        self.transforms = transforms
        self.tt = ToTensor()

    def __getitem__(self, idx):
        file_seq = self.df['file_seq'][idx]
        img_pair = []

        for file in file_seq:
            img = imageio.imread(os.path.join(self.prefix, self.df['path'][idx], file))
            img_pair.append(img)

        if self.transforms:
            seed = np.random.randint(np.iinfo(np.int32).max) # set random seed for numpy
            for i in range(len(img_pair)):
                imgaug.seed(seed)
                img_pair[i] = np.ascontiguousarray(self.transforms(img_pair[i]))

        # apply random edge homography
        reh = self.reh(img_pair[1], idx)

        img_crp = self.reh.crop(img_pair[0], reh['uv'])
        wrp_crp = reh['wrp_crp']

        for i in range(len(img_pair)):
            img_pair[i] = self.tt(img_pair[i])

        img_crp = self.tt(img_crp)
        wrp_crp = self.tt(wrp_crp)

        return {
            'img_pair': img_pair,
            'img_crp': img_crp,
            'wrp_crp': wrp_crp,
            'uv': reh['uv'], 
            'duv': reh['duv'], 
            'H': reh['H']
        }

    def __len__(self):
        return len(self.df)
