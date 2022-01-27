import os
import imgaug
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Callable, List


class ImageSequenceDataset(Dataset):
    r"""Reads an images sequence from a image database.

    Args:
        df (pd.DataFrame): Pandas dataframe, must contain {'folder': , 'file': , 'vid': , 'frame': }
        prefix (str): Path to database e.g. </path/to/database>/df.folder/df.file
        seq_len (int): Sequence length to sample images from, sequence length of 1 corresponds to static images, sequence length of 2 corresponds to neighboring images
        frame_increment (int): Sample every nth frame.
        frames_between_clips (int): Offset between initial frames of subsequent clips.
        transforms (Callable): Callable tranforms for augmenting sequences
        load_images (bool): Whether to return untransformed images
        seeds (bool): Seeds for deterministic output, e.g. for test set

    Returns:
        img_seq (torch.Tensor): Images shape NxCxHxW
        img_seq_transformed (torch.Tensor): Transformed images shape NxCxHxW
        idcs (List[int]): Frame indices
        vid_idx (int): Video index
    """
    def __init__(self,
        df: pd.DataFrame,
        prefix: str,
        seq_len: int=1,
        frame_increment: int=5,
        frames_between_clips: int=1,
        transforms: List[Callable]=None, 
        load_images: bool=True,
        seeds: bool=False
    ):
        self._df = df.sort_values(['vid', 'frame']).reset_index(drop=True)
        self._prefix = prefix
        self._seq_len = seq_len
        self._frame_increment = frame_increment
        self._frames_between_clips = frames_between_clips
        self._transforms = transforms
        self._load_images = load_images
        self._seeds = seeds
        self._idcs = self._filterFeasibleSequenceIndices(
            self._df, col='vid',
            seq_len=self._seq_len,
            frame_increment=self._frame_increment,
            frames_between_clips=self._frames_between_clips
        )

    @property
    def seq_len(self):
        return self._seq_len
    
    @seq_len.setter
    def seq_len(self, seq_len: int):
        self._seq_len = seq_len

    @property
    def frame_increment(self):
        return self._frame_increment

    frame_increment.setter
    def frame_increment(self, frame_increment: int):
        self.frame_increment = frame_increment

    def __getitem__(self, idx):
        # set seed if desired
        if self._seeds:
            seed = idx
        else:
            seed = random.randint(0, np.iinfo(np.int32).max)  # set random seed for numpy

        img_seq = []
        img_seq_transformed = []

        idcs = self._idcs[idx] + np.arange(self._seq_len)*self._frame_increment

        file_seq = self._df.loc[idcs]
        for _, row in file_seq.iterrows():
            img = np.load(os.path.join(self._prefix, row.folder, row.file))

            if self._load_images:
                img_seq.append(img)

            # transform image sequences
            if self._transforms:
                imgaug.seed(seed)
                img_seq_transformed.append(self._transforms(img))
            else:
                img_seq_transformed.append(img)

        if self._load_images:
            img_seq = torch.from_numpy(img_seq)
            img_seq = np.stack(img_seq).transpose(0,3,1,2)  # NxHxWxC -> NxCxHxW
        img_seq_transformed = torch.from_numpy(img_seq_transformed)
        img_seq_transformed = np.stack(img_seq_transformed).transpose(0,3,1,2)  # NxHxWxC -> NxCxHxW

        if self._load_images:
            return img_seq, img_seq_transformed, idcs, file_seq.vid.iloc[0]
        return img_seq_transformed, idcs, file_seq.vid.iloc[0]

    def __len__(self):
        return len(self._idcs)

    def _filterFeasibleSequenceIndices(self, 
        df: pd.DataFrame,
        col: str='vid',
        seq_len: int=2,
        frame_increment: int=1,
        frames_between_clips: int=1
    ) -> pd.DataFrame:
        grouped_df = df.groupby(col)
        return grouped_df.apply(
            lambda x: x.iloc[:len(x) - (seq_len - 1)*frame_increment:frames_between_clips]  # get indices [0, length - (seq_len - 1)]
        ).index.get_level_values(1)  # return 2nd values of pd.MultiIndex


class ImageSequenceDuvDataset(Dataset):
    r"""Reads an images sequence from a image database.

    Args:
        df (pd.DataFrame): Pandas dataframe, must contain {'folder': , 'file': , 'vid': , 'frame': , 'duv': }
        prefix (str): Path to database e.g. </path/to/database>/df.folder/df.file
        seq_len (int): Sequence length to sample images from, sequence length of 1 corresponds to static images, sequence length of 2 corresponds to neighboring images
        frame_increment (int): Sample every nth frame. Careful! Has to equal frame increment in df.
        frames_between_clips (int): Offset between initial frames of subsequent clips.
        transforms (Callable): Callable tranforms for augmenting sequences
        load_images (bool): Whether to load images
        seeds (bool): Seeds for deterministic output, e.g. for test set

    Returns:
        img_seq (torch.Tensor): Images shape NxCxHxW (if load_images)
        duv_seq (torch.Tensor): Duvs shape Nx4x2
        idcs (List[int]): Frame indices
        vid_idx (int): Video index
    """
    def __init__(self,
        df: pd.DataFrame,
        prefix: str,
        seq_len: int=1,
        frame_increment: int=5,
        frames_between_clips: int=1,
        transforms: List[Callable]=None, 
        load_images: bool=True,
        seeds: bool=False
    ):
        self._df = df.sort_values(['vid', 'frame']).reset_index(drop=True)
        self._prefix = prefix
        self._seq_len = seq_len
        self._frame_increment = frame_increment
        self._frames_between_clips = frames_between_clips
        self._transforms = transforms
        self._load_images = load_images
        self._seeds = seeds
        self._idcs = self._filterFeasibleSequenceIndices(
            self._df, col='vid',
            seq_len=self._seq_len,
            frame_increment=self._frame_increment,
            frames_between_clips=self._frames_between_clips
        )

    @property
    def seq_len(self):
        return self._seq_len
    
    @seq_len.setter
    def seq_len(self, seq_len: int):
        self._seq_len = seq_len

    @property
    def frame_increment(self):
        return self._frame_increment

    frame_increment.setter
    def frame_increment(self, frame_increment: int):
        self.frame_increment = frame_increment

    def __getitem__(self, idx):
        # set seed if desired
        if self._seeds:
            seed = idx
        else:
            seed = random.randint(0, np.iinfo(np.int32).max)  # set random seed for numpy

        img_seq = []
        duv_seq = []

        idcs = self._idcs[idx] + np.arange(self._seq_len)*self._frame_increment

        file_seq = self._df.loc[idcs]
        for _, row in file_seq.iterrows():
            if self._load_images:
                img = np.load(os.path.join(self._prefix, row.folder, row.file))

                # transform image sequences
                if self._transforms:
                    imgaug.seed(seed)
                    img_seq.append(self._transforms(img))
                else:
                    img_seq.append(img)

            duv_seq.append(np.array(row.duv))

        if self._load_images:
            img_seq = np.stack(img_seq).transpose(0,3,1,2)  # NxHxWxC -> NxCxHxW
            img_seq = torch.from_numpy(img_seq)
        duv_seq = torch.from_numpy(np.stack(duv_seq))

        if self._load_images:
            return img_seq, duv_seq, idcs, file_seq.vid.iloc[0]
        return duv_seq, idcs, file_seq.vid.iloc[0]

    def __len__(self):
        return len(self._idcs)

    def _filterFeasibleSequenceIndices(self, 
        df: pd.DataFrame,
        col: str='vid',
        seq_len: int=2,
        frame_increment: int=1,
        frames_between_clips: int=1
    ) -> pd.DataFrame:
        grouped_df = df.groupby(col)
        return grouped_df.apply(
            lambda x: x.iloc[:len(x) - (seq_len - 1)*frame_increment:frames_between_clips]  # get indices [0, length - (seq_len - 1)]
        ).index.get_level_values(1)  # return 2nd values of pd.MultiIndex


class ImageSequenceDatasetSequenceDf(Dataset):
    r"""Reads an images sequence from a image database.

    Args:
        df (pd.DataFrame): Pandas dataframe, must contain {'file_seq': [frame0.png, frame1.png], 'path': 'path/to/frames'}
        prefix (str): Path to database e.g. </path/to/database>/path/to/frames/frame.png
        transforms (callable): Transforms to be applied

    Returns:
        img_seq (list of Torch.tensor): Sequence of images of shape CxHxW
    
    Note:
        Legacy code.
    """
    def __init__(self, df: pd.DataFrame, prefix: str, transforms: Callable=None):
        self._df = df
        self._prefix = prefix
        self._transforms = transforms

    def __getitem__(self, idx):
        file_seq = self._df['file_seq'][idx]
        img_seq = []

        for file in file_seq:
            img = np.load(os.path.join(self._prefix, self._df['path'][idx], file))
            if self._transforms:
                img = self._transforms(img)
            img_seq.append(img)

        return img_seq

    def __len__(self):
        return len(self._df)


if __name__ == '__main__':
    import os
    import numpy as np
    import pandas as pd
    from dotmap import DotMap
    import cv2

    from utils.io import load_yaml

    server = 'local'
    server = DotMap(load_yaml('config/servers.yml')[server])
    prefix = os.path.join(server.database.location, 'camera_motion_separated_npy/without_camera_motion')
    pkl_name = 'light_log_without_camera_motion.pkl'
    df = pd.read_pickle(os.path.join(prefix, pkl_name))
    seq_len = 10

    col = 'vid'
    grouped_df = df.groupby(col)
    idcs = grouped_df.apply(lambda x: x.iloc[:len(x) - (seq_len - 1)]).index.get_level_values(1)
    print(len(idcs))

    dummy_idx = 0
    print(idcs[dummy_idx])

    # sample
    print(df.loc[idcs[dummy_idx]])

    # random index
    seq_idcs = idcs[dummy_idx] + np.arange(seq_len)
    print(seq_idcs)

    # sample
    file_seq = df.loc[seq_idcs]
    print(file_seq)
    print(file_seq.vid.iloc[0])

    # load
    img_seq = []
    for _, row in file_seq.iterrows():
        img = np.load(os.path.join(prefix, row.folder, row.file))
        img_seq.append(img)
        cv2.imshow('img', img)
        cv2.waitKey()

    img_seq = np.stack(img_seq).transpose(0,3,1,2)
    img_seq = torch.from_numpy(img_seq)
