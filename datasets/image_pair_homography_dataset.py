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
        df (pd.DataFrame): Pandas dataframe, must contain {'folder': , 'file': , 'vid': , 'frame': }
        prefix (str): Path to database e.g. </path/to/database>/df.folder/df.file
        rho (int): Image edges are randomly perturbed within [-rho, rho]
        crp_shape (list of int): Shape of cropped image
        p0 (float): Chance for homography being identity
        seq_len (int): Sequence length to sample images from, sequence length of 1 corresponds to static images, sequence length of 2 corresponds to neighboring images
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
        transforms: Callable=None, 
        seeds: List[np.int32]=None, 
        return_img_pair: bool=True
    ) -> None:
        if seeds:
            if (len(df) != len(seeds)):
                raise Exception('In ImagePairHomographyDataset: Length of dataframe must equal length of seeds.')
        
        self._df = df
        self._prefix = prefix
        self._rho = rho
        self._reh = RandomEdgeHomography(rho=rho, crp_shape=crp_shape, p0=p0, homography_return=HOMOGRAPHY_RETURN.DATASET, seeds=seeds)
        if seq_len < 1:
            raise ValueError('Sequence length {} must be greater or equal 1.'.format(seq_len))
        self._seq_len = seq_len
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
            seed = np.random.randint(np.iinfo(np.int32).max)  # set random seed for numpy

        # randomly sample image pair
        np.random.seed(seed)
        idcs = self._idcs[idx] + np.random.choice(np.arange(self._seq_len), 2, replace=(self._seq_len == 1))  # static if self._seq_len = 1
        np.random.seed(None)

        file_pair = self._df.loc[idcs]

        for _, row in file_pair.iterrows():
            img = imageio.imread(os.path.join(self._prefix, row.folder, row.file))
            img_pair.append(img)

        if self._transforms:
            for i in range(len(img_pair)):
                imgaug.seed(seed)
                img_pair[i] = np.ascontiguousarray(self._transforms(img_pair[i]))

        # apply random edge homography
        self._reh.seed_idx = idx
        reh = self._reh(img_pair[1])

        img_crp = self._reh.crop(img_pair[0], reh['uv'])
        wrp_crp = reh['wrp_crp']

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


class ImagePairHomographyDatasetSequenceDf(Dataset):
    r"""Takes two images, supposedly from two time steps, and warps the second. Returns crops of both. 
    Implements the method described by DeTone et al. in https://arxiv.org/pdf/1606.03798.pdf.

    Args:
        df (pd.DataFrame): Pandas dataframe, must contain {'file_seq': [frame0.png, frame1.png], 'path': 'path/to/frames'}
        prefix (str): Path to database e.g. </path/to/database>/path/to/frames/frame.png
        rho (int): Image edges are randomly perturbed within [-rho, rho]
        crp_shape (list of int): Shape of cropped image
        p0 (float): Chance for homography being identity
        rnd_time_sample (bool): Whether or not to sample time randomly
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

    Notes:
        Legacy code.
    """
    def __init__(self, 
        df: pd.DataFrame, 
        prefix: str, 
        rho: int, 
        crp_shape: List[int],
        p0: float=0., 
        rnd_time_sample: bool=True, 
        transforms: Callable=None, 
        seeds: List[np.int32]=None, 
        return_img_pair: bool=True
    ) -> None:
        if seeds:
            if (len(df) != len(seeds)):
                raise Exception('In ImagePairHomographyDatasetSequenceDf: Length of dataframe must equal length of seeds.')
        
        self._df = df
        self._prefix = prefix   
        self._reh = RandomEdgeHomography(rho=rho, crp_shape=crp_shape, p0=p0, homography_return=HOMOGRAPHY_RETURN.DATASET, seeds=seeds)
        self._rnd_time_sample = rnd_time_sample
        self._transforms = transforms
        self._seeds = seeds
        self._return_image_pair = return_img_pair
        self._tt = ToTensor()

    def __getitem__(self, idx):
        file_seq = self._df['file_seq'][idx]
        img_pair = []

        # set seed if desired
        if self._seeds:
            seed = self._seeds[idx]
        else:
            seed = np.random.randint(np.iinfo(np.int32).max)  # set random seed for numpy

        # randomly sample image pair
        if self._rnd_time_sample:
            np.random.seed(seed)
            file_pair = np.random.choice(file_seq, 2)
            np.random.seed(None)
        else:
            file_pair = [file_seq[0], file_seq[1]]

        for file in file_pair:
            img = imageio.imread(os.path.join(self._prefix, self._df['path'][idx], file))
            img_pair.append(img)

        if self._transforms:
            for i in range(len(img_pair)):
                imgaug.seed(seed)
                img_pair[i] = np.ascontiguousarray(self._transforms(img_pair[i]))

        # apply random edge homography
        self._reh.seed_idx = idx
        reh = self._reh(img_pair[1])

        img_crp = self._reh.crop(img_pair[0], reh['uv'])
        wrp_crp = reh['wrp_crp']

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
        return len(self._df)


if __name__ == '__main__':
    import os
    import numpy as np
    import pandas as pd
    from dotmap import DotMap
    import imageio
    import cv2

    from utils.io import load_yaml

    server = 'local'
    server = DotMap(load_yaml('configs/servers.yml')[server])
    prefix = os.path.join(server.database.location, 'camera_motion_separated_png/without_camera_motion')

    pkl_name = 'light_log_without_camera_motion.pkl'
    df = pd.read_pickle(os.path.join(prefix, pkl_name))
    seq_len = 25  # static if seq_len = 1

    col = 'vid'
    grouped_df = df.groupby(col)
    idcs = grouped_df.apply(lambda x: x.iloc[seq_len-1:len(x) - (seq_len - 1)]).index.get_level_values(1)
    print(len(idcs))

    dummy_idx = 0
    print(idcs[dummy_idx])

    # sample
    print(df.loc[idcs[dummy_idx]])

    # random index
    rnd_idcs = idcs[dummy_idx] + np.random.choice(np.arange(seq_len), 2, replace=(seq_len == 1))
    print(rnd_idcs)

    # sample
    file_pair = df.loc[rnd_idcs]
    print(file_pair)

    # load
    for _, row in file_pair.iterrows():
        img = imageio.imread(os.path.join(prefix, row.folder, row.file))
        cv2.imshow('img', img)
        cv2.waitKey()
