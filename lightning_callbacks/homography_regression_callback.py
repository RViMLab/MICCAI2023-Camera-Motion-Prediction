import importlib
from typing import Any

import pytorch_lightning as pl
import torch

from utils import frame_pairs


class HomographyRegressionCallback(pl.Callback):
    r"""The HomographyRegressionCallback computes a homography for given image pairs and
    appends it to the batch.
    """

    def __init__(
        self, package: str, module: str, device: str, checkpoint_path: str, **kwargs
    ) -> None:
        super().__init__()
        print(
            f"HomographyRegressionCallback: Loading homography regression from {checkpoint_path}"
        )
        self._homography_regression = getattr(
            importlib.import_module(package), module
        ).load_from_checkpoint(checkpoint_path=checkpoint_path, **kwargs)
        print("Done.")
        self._device = device
        self._homography_regression.to(self._device)

        self._freeze()

    def _freeze(self):
        self._homography_regression = self._homography_regression.eval()
        self._homography_regression.freeze()

    def _regress_homography(self, imgs: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = imgs.shape
        imgs, wrps = frame_pairs(imgs, T - 1)
        imgs, wrps = imgs.float() / 255.0, wrps.float() / 255.0

        with torch.no_grad():
            imgs, wrps = imgs.view(-1, C, H, W), wrps.view(-1, C, H, W)
            duv = self._homography_regression(imgs, wrps)
            duv = duv.view(B, 1, 4, 2)

        return duv

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        imgs = batch[0]
        duv = self._regress_homography(imgs)
        batch.append(duv)
        return super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        imgs = batch[0]
        duv = self._regress_homography(imgs)
        batch.append(duv)
        return super().on_validation_batch_start(
            trainer, pl_module, batch, batch_idx, dataloader_idx
        )

    def on_test_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        imgs = batch[0]
        duv = self._regress_homography(imgs)
        batch.append(duv)
        return super().on_test_batch_start(
            trainer, pl_module, batch, batch_idx, dataloader_idx
        )


if __name__ == "__main__":
    import os

    from utils.io import load_yaml

    # load configs
    server = "local"
    server = load_yaml("config/servers.yml")[server]

    in_logging_location_prefix = "ae_cai/resnet/48/25/34/version_0"
    config = load_yaml(
        os.path.join(
            server["logging"]["location"], in_logging_location_prefix, "config.yml"
        )
    )
    checkpoint_path = os.path.join(
        server["logging"]["location"],
        in_logging_location_prefix,
        "checkpoints/epoch=99-step=47199.ckpt",
    )

    # create homography regressor
    homography_regression_callback = HomographyRegressionCallback(
        package="lightning_modules",
        module="DeepImageHomographyEstimationModuleBackbone",
        device="cpu",  # for debugging
        checkpoint_path=checkpoint_path,
        **config["model"],
    )

    # add sample data loader
    import pandas as pd

    from lightning_data_modules import ImageSequenceDataModule

    data_prefix = os.path.join(
        server["database"]["location"], "21_11_25_first_test_data_frames"
    )
    df = pd.read_pickle(os.path.join(data_prefix, "test_train_log.pkl"))

    dm = ImageSequenceDataModule(
        df=df,
        prefix=data_prefix,
        train_split=0.8,
        batch_size=1,
        num_workers=1,
        random_state=42,
        tolerance=0.2,
        seq_len=10,
        frame_increment=2,
        frames_between_clips=20,
        load_images=True,
    )

    dm.setup()
    train_dl = dm.train_dataloader()

    for batch in train_dl:
        homography_regression_callback._regress_homography(batch[0])
