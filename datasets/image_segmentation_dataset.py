import os
import imgaug
import numpy as np
from numpy.lib.npyio import load
import pandas as pd
from typing import Callable, List
from torch.utils.data import Dataset


class ImageSegmentationDataset(Dataset):
    r"""Samples image and corresponding segmentation from a dataframe and applies transforms to it.

    Args:
        df (pd.DataFrame): Pandas dataframe of the form {'image': {'file': , 'folder': }, 'segmentation': {'file': , 'folder': }}
        prefix (str): Prefix to database e.g. </path/to/database>/df.image['folder']/df.image['file']
        image_transforms (Callable): Transforms to be applied to the image
        spatial_transforms (Callable): Spatial transforms to be applied to image and segmentation
        seeds (List[int]): Seeds for deterministic output, e.g. for test set
    """
    def __init__(
        self,
        df: pd.DataFrame, 
        prefix: str, 
        image_transforms: Callable=None, 
        spatial_transforms: Callable=None,
        seeds: List[int]=None
    ) -> None:
        super().__init__()

        if seeds:
            if (len(df) != len(seeds)):
                raise Exception('In ImageSegmentationDataset: Length of dataframe must equal length of seeds.')

        self._df = df
        self._prefix = prefix
        self._image_transforms = image_transforms
        self._spatial_transforms = spatial_transforms
        self._seeds = seeds

    def __getitem__(self, idx):
        # load image and segmentation
        img = np.load(os.path.join(self._prefix, self._df.iloc[idx].image['folder'], self._df.iloc[idx].image['file']))
        seg = np.load(os.path.join(self._prefix, self._df.iloc[idx].segmentation['folder'], self._df.iloc[idx].segmentation['file']))

        if self._seeds:
            seed = self._seeds[idx]
        else:
            seed = np.random.randint(0, np.iinfo(np.int32).max)

        # apply transforms
        if self._spatial_transforms:
            imgaug.seed(seed)
            img = self._spatial_transforms(img)
            imgaug.seed(seed)
            seg = self._spatial_transforms(seg)

        if self._image_transforms:
            imgaug.seed(seed)
            img = self._image_transforms(img)

        return img, seg
 
    def __len__(self):
        return len(self._df)


if __name__ == '__main__':
    import os
    import sys
    sys.path.append('..')
    import pandas as pd

    from utils.transforms import dictListToAugment
    from utils.io import load_yaml

    configs = load_yaml('config/boundary_segmentation.yml')

    spatial_transforms = configs['data']['spatial_transforms']
    image_transforms = configs['data']['image_transforms']

    spatial_transforms = dictListToAugment(spatial_transforms)
    image_transforms = dictListToAugment(image_transforms)

    prefix = '/media/martin/Samsung_T5/data/endoscopic_data/boundary_segmentation'

    df = pd.read_pickle(os.path.join(prefix, 'light_log.pkl'))
    print(df)

    ds = ImageSegmentationDataset(df, prefix, spatial_transforms=spatial_transforms, image_transforms=image_transforms)

    for i in range(1000):

        img, seg = ds.__getitem__(i)

        import cv2

        cv2.imshow('img', img)
        cv2.imshow('seg', seg)

        cv2.waitKey()

    # # generate df
    # import os
    # import sys
    # sys.path.append('..')

    # from utils.io import recursive_scan2df

    # def sortFunc(e):
    #     id = int(e['file'].split('.')[0])
    #     return id

    # prefix = '/media/martin/Samsung_T5/data/endoscopic_data/boundary_segmentation'

    # df = recursive_scan2df(prefix, '.npy')
    

    # images = [{'folder': row.folder, 'file': row.file} for _, row in df[df.folder == 'frame'].iterrows()]
    # segmentations = [{'folder': row.folder, 'file': row.file} for _, row in df[df.folder == 'segmentation'].iterrows()]

    # images.sort(key=sortFunc)
    # segmentations.sort(key=sortFunc)

    # log_df = pd.DataFrame({'image': images, 'segmentation': segmentations})
    # log_df.to_pickle(os.path.join(prefix, 'light_log.pkl'))
    # log_df.to_csv(os.path.join(prefix, 'light_log.csv'))




    
