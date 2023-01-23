import importlib
import os

import pytorch_lightning as pl
import torch

import lightning_modules


class ConvHomographyPredictorModule(pl.LightningModule):
    def __init__(
        self,
        predictor: dict,
        optimizer: dict,
        loss: dict,
    ):
        super().__init__()
        self._predictor = getattr(
            importlib.import_module(predictor["module"]), predictor["name"]
        )(**predictor["kwargs"])

        self._optimizer = getattr(
            importlib.import_module(optimizer["module"]), optimizer["name"]
        )(params=self.parameters(), **optimizer["kwargs"])

        self._loss = getattr(importlib.import_module(loss["module"]), loss["name"])(
            **loss["kwargs"]
        )

        self._homography_regression = None

    def inject_homography_regression(
        self, homography_regression: dict, homography_regression_prefix: str
    ):
        # load trained homography regression model
        self._homography_regression = getattr(
            lightning_modules, homography_regression["lightning_module"]
        ).load_from_checkpoint(
            checkpoint_path=os.path.join(
                homography_regression_prefix,
                homography_regression["path"],
                homography_regression["checkpoint"],
            ),
            **homography_regression["model"]
        )
        self._homography_regression = self._homography_regression.eval()
        self._homography_regression.freeze()

    def on_train_epoch_start(self):
        if self._homography_regression:
            self._homography_regression = self._homography_regression.eval()
            self._homography_regression.freeze()

    def configure_optimizers(self):
        return self._optimizer

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self._predictor(imgs)

    def training_step(self, batch, batch_idx):
        (
            tf_imgs,
            duvs_reg,
            frame_idcs,
            vid_idcs,
        ) = batch  # transformed images and four point homography
        tf_imgs = tf_imgs.float() / 255.0
        duvs_reg = duvs_reg.float()

        B, T, C, H, W = tf_imgs.shape
        tf_imgs = tf_imgs.view(B, T * C, H, W)
        duv = self(tf_imgs)
        duv = duv.view(B, 4, 2)

        loss = self._loss(duv.view(-1, 2), duvs_reg[:, -1].view(-1, 2))

        return {
            "loss": loss.mean(),
        }

    def validation_step(self, batch, batch_idx):
        (
            tf_imgs,
            duvs_reg,
            frame_idcs,
            vid_idcs,
        ) = batch  # transformed images and four point homography
        tf_imgs = tf_imgs.float() / 255.0
        duvs_reg = duvs_reg.float()

        B, T, C, H, W = tf_imgs.shape
        tf_imgs = tf_imgs.view(B, T * C, H, W)
        duv = self(tf_imgs)
        duv = duv.view(B, 4, 2)

        loss = self._loss(duv.view(-1, 2), duvs_reg[:, -1].view(-1, 2))

        self.log("val/loss", loss.mean())

    def test_step(self, batch, batch_idx):
        pass
