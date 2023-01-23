import importlib
import os

import pytorch_lightning as pl
import torch

import lightning_modules
from utils import frame_pairs
from utils.viz import create_blend_from_four_point_homography, yt_alpha_blend


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
        self._train_logged = False
        self._val_logged = False

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
            imgs,
            tf_imgs,
            frame_idcs,
            vid_idcs,
        ) = batch  # transformed images and four point homography
        B, T, C, H, W = imgs.shape
        imgs, wrps = frame_pairs(imgs, T-1)
        imgs, wrps = imgs.float() / 255.0, wrps.float() / 255.0
        tf_imgs = tf_imgs.float() / 255.0

        with torch.no_grad():
            imgs, wrps = imgs.view(-1, C, H, W), wrps.view(-1, C, H, W)
            duv_reg = self._homography_regression(imgs, wrps)
            duv_reg = duv_reg.view(B, 1, 4, 2)

        imgs = tf_imgs[:, :-1]
        imgs = imgs.view(B, -1, H, W)
        duv_pred = self(imgs)
        duv_pred = duv_pred.view(B, 1, 4, 2)

        loss = self._loss(duv_pred.view(-1, 2), duv_reg.reshape(-1, 2))
        norm = self._loss(duv_pred.view(-1, 2), torch.zeros_like(duv_pred).view(-1, 2))

        if not self._train_logged:
            self._train_logged = True
            blend_identity = yt_alpha_blend(
                tf_imgs[0, 0].unsqueeze(0),
                tf_imgs[0, T-1].unsqueeze(0),
            )
            blend_reg = create_blend_from_four_point_homography(
                tf_imgs[0, 0].unsqueeze(0),
                tf_imgs[0, T-1].unsqueeze(0),
                duv_reg[0],
            )
            blend_pred = create_blend_from_four_point_homography(
                tf_imgs[0, 0].unsqueeze(0), tf_imgs[0, T-1].unsqueeze(0), duv_pred[0]
            )

            self.logger.experiment.add_images(
                "train/transformed_imgs", tf_imgs[0], self.global_step
            )
            self.logger.experiment.add_images(
                "train/blend/identity", blend_identity, self.global_step
            )
            self.logger.experiment.add_images(
                "train/blend/regressed", blend_reg, self.global_step
            )
            self.logger.experiment.add_images(
                "train/blend/predicted", blend_pred, self.global_step
            )

        self.log("train/loss", loss.mean())
        self.log("train/norm", norm.mean())

        return {
            "loss": loss.mean(),
        }

    def validation_step(self, batch, batch_idx):
        (
            imgs,
            tf_imgs,
            frame_idcs,
            vid_idcs,
        ) = batch  # transformed images and four point homography
        B, T, C, H, W = imgs.shape
        imgs, wrps = frame_pairs(imgs, T-1)
        imgs, wrps = imgs.float() / 255.0, wrps.float() / 255.0
        tf_imgs = tf_imgs.float() / 255.0

        with torch.no_grad():
            imgs, wrps = imgs.view(-1, C, H, W), wrps.view(-1, C, H, W)
            duv_reg = self._homography_regression(imgs, wrps)
            duv_reg = duv_reg.view(B, 1, 4, 2)

        imgs = tf_imgs[:, :-1]
        imgs = imgs.view(B, -1, H, W)
        duv_pred = self(imgs)
        duv_pred = duv_pred.view(B, 1, 4, 2)

        loss = self._loss(duv_pred.view(-1, 2), duv_reg.reshape(-1, 2))
        norm = self._loss(duv_pred.view(-1, 2), torch.zeros_like(duv_pred).view(-1, 2))

        if not self._val_logged:
            self._val_logged = True
            blend_identity = yt_alpha_blend(
                tf_imgs[0, 0].unsqueeze(0),
                tf_imgs[0, T-1].unsqueeze(0),
            )
            blend_reg = create_blend_from_four_point_homography(
                tf_imgs[0, 0].unsqueeze(0),
                tf_imgs[0, T-1].unsqueeze(0),
                duv_reg[0],
            )
            blend_pred = create_blend_from_four_point_homography(
                tf_imgs[0, 0].unsqueeze(0), tf_imgs[0, T-1].unsqueeze(0), duv_pred[0]
            )

            self.logger.experiment.add_images(
                "val/transformed_imgs", tf_imgs[0], self.global_step
            )
            self.logger.experiment.add_images(
                "val/blend/identity", blend_identity, self.global_step
            )
            self.logger.experiment.add_images(
                "val/blend/regressed", blend_reg, self.global_step
            )
            self.logger.experiment.add_images(
                "val/blend/predicted", blend_pred, self.global_step
            )

        self.log("val/loss", loss.mean())
        self.log("val/norm", norm.mean())

    def test_step(self, batch, batch_idx):
        pass

    def on_train_epoch_end(self) -> None:
        self._train_logged = False
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        self._val_logged = False
        return super().on_validation_epoch_end()
