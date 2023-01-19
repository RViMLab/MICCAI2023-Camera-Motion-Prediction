from typing import List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader

from datasets import ImageHomographyMaskDataset
from utils import dict_list_to_augment


class ImageHomographyMaskDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataframe: str,
        prefix: str,
        rho: int,
        train_split: float,
        batch_size: int,
        num_workers: int = 2,
        random_state: int = 42,
        tolerance: float = 0.05,
        train_transforms: List[dict] = None,
        val_transforms: List[dict] = None,
        test_transforms: List[dict] = None,
    ) -> None:
        super().__init__()

        df = pd.read_pickle(f"{prefix}/{dataframe}")

        # train/test
        self._train_df = df[df.train == True]
        self._test_df = df[df.train == False]

        # further split train into train and validation set
        unique_vid = self._train_df.vid.unique()

        train_vid, val_vid = train_test_split(
            unique_vid, train_size=train_split, random_state=random_state
        )

        self._val_df = self._train_df[
            self._train_df.vid.apply(lambda x: x in val_vid)
        ].reset_index()
        self._train_df = self._train_df[
            self._train_df.vid.apply(lambda x: x in train_vid)
        ].reset_index()

        # assert if fraction off
        fraction = len(self._val_df) / (len(self._train_df) + len(self._val_df))
        assert np.isclose(
            fraction, 1 - train_split, atol=tolerance
        ), "Train set fraction {:.3f} not close enough to (1 - train_split) {} at tolerance {}".format(
            fraction, 1 - train_split, tolerance
        )

        self._train_tranforms = dict_list_to_augment(train_transforms)
        self._val_tranforms = dict_list_to_augment(val_transforms)
        self._test_tranforms = dict_list_to_augment(test_transforms)

        self._prefix = prefix
        self._rho = rho
        self._batch_size = batch_size
        self._num_workers = num_workers

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            self._train_ds = ImageHomographyMaskDataset(
                self._train_df,
                self._prefix,
                self._rho,
                self._train_tranforms,
                seeds=False,
            )
            self._val_ds = ImageHomographyMaskDataset(
                self._val_df, self._prefix, self._rho, self._val_tranforms, seeds=True
            )
        if stage == "test":
            self._test_ds = ImageHomographyMaskDataset(
                self._test_df, self._prefix, self._rho, self._test_tranforms, seeds=True
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self._train_ds,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self._val_ds,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self._test_ds,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            drop_last=False,
            pin_memory=True,
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from kornia import tensor_to_image

    dm = ImageHomographyMaskDataModule(
        dataframe="22_11_09_deep_log_pre_processed_test_train_no_nan.pkl",
        prefix="/media/martin/Samsung_T5/data/endoscopic_data/cholec80_frames",
        train_split=0.8,
        batch_size=1,
        num_workers=0,
        tolerance=0.2,
    )

    dm.setup()
    train_dl = dm.train_dataloader()
    for imgs in train_dl:
        img = imgs[0]
        img = tensor_to_image(img)
        print(img.shape)
        print(img.shape)
        plt.imshow(img)
        plt.show()
        break
