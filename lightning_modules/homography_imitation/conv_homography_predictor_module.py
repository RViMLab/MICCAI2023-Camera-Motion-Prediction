import importlib

import pytorch_lightning as pl
import torch

from utils.processing import TaylorHomographyPrediction, frame_pairs
from utils.viz import create_blend_from_four_point_homography, yt_alpha_blend


class ConvHomographyPredictorModule(pl.LightningModule):
    def __init__(
        self,
        predictor: dict,
        optimizer: dict,
        loss: dict,
        scheduler: dict,
        preview_horizon: int = 1,
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

        self._scheduler = None
        if scheduler:
            self._scheduler = getattr(
                importlib.import_module(scheduler["module"]), scheduler["name"]
            )(optimizer=self._optimizer, **scheduler["kwargs"])

        self._preview_horizon = preview_horizon
        self._taylor_1st_order = TaylorHomographyPrediction(order=1)
        self._taylor_2nd_order = TaylorHomographyPrediction(order=2)
        self._log_nth_epoch = 1

    def configure_optimizers(self):
        if self._scheduler:
            return {
                "optimizer": self._optimizer,
                "lr_scheduler": self._scheduler,
                "monitor": "val/loss",
            }
        return self._optimizer

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self._predictor(imgs).view(-1, 4, 2)

    def training_step(self, batch, batch_idx):
        (
            imgs,
            tf_imgs,
            frame_idcs,
            vid_idcs,
            duv_reg,  # added through HomographyRegressionCallback
        ) = batch  # transformed images and four point homography
        B, T, C, H, W = imgs.shape
        tf_imgs = tf_imgs.float() / 255.0

        imgs = tf_imgs[:, : -self._preview_horizon]
        imgs = imgs.view(B, -1, H, W)
        duv_pred = self(imgs)

        loss = self._loss(
            duv_pred.view(-1, 2), duv_reg[:, -self._preview_horizon :].reshape(-1, 2)
        )
        norm_reg = self._loss(
            duv_reg[:, -self._preview_horizon :].reshape(-1, 2),
            torch.zeros_like(duv_pred).view(-1, 2),
        )
        norm_pred = self._loss(
            duv_pred.view(-1, 2), torch.zeros_like(duv_pred).view(-1, 2)
        )

        if self.current_epoch % self._log_nth_epoch == 0 and batch_idx == 0:
            tf_imgs, tf_wrps = frame_pairs(tf_imgs)
            blend_identity = yt_alpha_blend(
                tf_imgs[0, -self._preview_horizon :],
                tf_wrps[0, -self._preview_horizon :],
            )
            blend_reg = create_blend_from_four_point_homography(
                tf_imgs[0, -self._preview_horizon :],
                tf_wrps[0, -self._preview_horizon :],
                duv_reg[0, -self._preview_horizon :],
            )
            blend_pred = create_blend_from_four_point_homography(
                tf_imgs[0, -self._preview_horizon :],
                tf_wrps[0, -self._preview_horizon :],
                duv_pred[0],
            )

            # self.logger.experiment.add_images(
            #     "train/transformed_imgs", tf_imgs[0], self.global_step
            # )
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
        self.log("train/norm_reg", norm_reg.mean())
        self.log("train/norm_pred", norm_pred.mean())

        return {
            "loss": loss.mean(),
            "duv_reg": duv_reg.detach().cpu().numpy(),
            "duv_pred": duv_pred.detach().cpu().numpy(),
        }

    def validation_step(self, batch, batch_idx):
        (
            imgs,
            tf_imgs,
            frame_idcs,
            vid_idcs,
            duv_reg,  # added through HomographyRegressionCallback
        ) = batch  # transformed images and four point homography
        B, T, C, H, W = imgs.shape
        tf_imgs = tf_imgs.float() / 255.0

        imgs = tf_imgs[:, : -self._preview_horizon]
        imgs = imgs.view(B, -1, H, W)
        duv_pred = self(imgs)

        duv_pred_taylor_1st_order = self._taylor_1st_order(duv_reg.cpu())[
            :, -self._preview_horizon :
        ].to(self.device)
        duv_pred_taylor_2nd_order = self._taylor_2nd_order(duv_reg.cpu())[
            :, -self._preview_horizon :
        ].to(self.device)

        loss = self._loss(
            duv_pred.view(-1, 2), duv_reg[:, -self._preview_horizon :].reshape(-1, 2)
        )
        loss_taylor_1st_order = self._loss(
            duv_pred_taylor_1st_order.reshape(-1, 2),
            duv_reg[:, -self._preview_horizon :].reshape(-1, 2),
        )
        loss_taylor_2nd_order = self._loss(
            duv_pred_taylor_2nd_order.reshape(-1, 2),
            duv_reg[:, -self._preview_horizon :].reshape(-1, 2),
        )
        norm_reg = self._loss(
            duv_reg[:, -self._preview_horizon :].reshape(-1, 2),
            torch.zeros_like(duv_pred).view(-1, 2),
        )
        norm_pred = self._loss(
            duv_pred.view(-1, 2), torch.zeros_like(duv_pred).view(-1, 2)
        )

        if self.current_epoch % self._log_nth_epoch == 0 and batch_idx == 0:
            tf_imgs, tf_wrps = frame_pairs(tf_imgs)
            blend_identity = yt_alpha_blend(
                tf_imgs[0, -self._preview_horizon :],
                tf_wrps[0, -self._preview_horizon :],
            )
            blend_reg = create_blend_from_four_point_homography(
                tf_imgs[0, -self._preview_horizon :],
                tf_wrps[0, -self._preview_horizon :],
                duv_reg[0, -self._preview_horizon :],
            )
            blend_pred = create_blend_from_four_point_homography(
                tf_imgs[0, -self._preview_horizon :],
                tf_wrps[0, -self._preview_horizon :],
                duv_pred[0],
            )

            # self.logger.experiment.add_images(
            #     "val/transformed_imgs", tf_imgs[0], self.global_step
            # )
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
        self.log("val/loss_taylor_1st_order", loss_taylor_1st_order.mean())
        self.log("val/loss_taylor_2nd_order", loss_taylor_2nd_order.mean())
        self.log(
            "val/loss_taylor_1st_order_minus_loss",
            (loss_taylor_1st_order - loss).mean(),
        )
        self.log(
            "val/loss_taylor_2nd_order_minus_loss",
            (loss_taylor_2nd_order - loss).mean(),
        )
        self.log("val/norm_reg", norm_reg.mean())
        self.log("val/norm_pred", norm_pred.mean())

    def test_step(self, batch, batch_idx):
        (
            imgs,
            tf_imgs,
            frame_idcs,
            vid_idcs,
            duv_reg,  # added through HomographyRegressionCallback
        ) = batch  # transformed images and four point homography
        B, T, C, H, W = imgs.shape
        imgs = imgs.float() / 255.0

        imgs = imgs[:, : -self._preview_horizon]
        imgs = imgs.view(B, -1, H, W)
        duv_pred = self(imgs)

        duv_pred_taylor_1st_order = self._taylor_1st_order(duv_reg.cpu())[
            :, -self._preview_horizon :
        ].to(self.device)
        duv_pred_taylor_2nd_order = self._taylor_2nd_order(duv_reg.cpu())[
            :, -self._preview_horizon :
        ].to(self.device)

        loss_taylor_1st_order = self._loss(
            duv_pred_taylor_1st_order.reshape(-1, 2),
            duv_reg[:, -self._preview_horizon :].reshape(-1, 2),
        )
        loss_taylor_2nd_order = self._loss(
            duv_pred_taylor_2nd_order.reshape(-1, 2),
            duv_reg[:, -self._preview_horizon :].reshape(-1, 2),
        )

        loss = self._loss(
            duv_pred.view(-1, 2), duv_reg[:, -self._preview_horizon :].reshape(-1, 2)
        )
        norm_reg = self._loss(
            duv_reg[:, -self._preview_horizon :].reshape(-1, 2),
            torch.zeros_like(duv_pred).view(-1, 2),
        )
        norm_pred = self._loss(
            duv_pred.view(-1, 2), torch.zeros_like(duv_pred).view(-1, 2)
        )

        self.log("test/loss", loss.mean(), on_epoch=True)
        self.log(
            "val/loss_taylor_1st_order", loss_taylor_1st_order.mean(), on_epoch=True
        )
        self.log(
            "val/loss_taylor_2nd_order", loss_taylor_2nd_order.mean(), on_epoch=True
        )
        self.log(
            "val/loss_taylor_1st_order_minus_loss",
            (loss_taylor_1st_order - loss).mean(),
            on_epoch=True,
        )
        self.log(
            "val/loss_taylor_2nd_order_minus_loss",
            (loss_taylor_2nd_order - loss).mean(),
            on_epoch=True,
        )
        self.log("test/norm_reg", norm_reg.mean(), on_epoch=True)
        self.log("test/norm_pred", norm_pred.mean(), on_epoch=True)
