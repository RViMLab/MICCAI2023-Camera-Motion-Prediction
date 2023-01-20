import importlib
import os
from typing import Any, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from kornia.geometry import warp_perspective
from pytorch_lightning.utilities.types import STEP_OUTPUT

import lightning_modules
from utils.processing import (
    TaylorHomographyPrediction,
    differentiate_duv,
    four_point_homography_to_matrix,
    frame_pairs,
    image_edges,
    integrate_duv,
)
from utils.viz import (
    create_blend_from_four_point_homography,
    uv_trajectory_figure,
    yt_alpha_blend,
)


class DuvLSTMModule(pl.LightningModule):
    def __init__(
        self,
        lstm_hidden_size: int = 512,
        lr: float = 1e-4,
        betas: List[float] = [0.9, 0.999],
        frame_stride: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters("lr", "betas")

        self._homography_regression = None

        # load model
        self._lstm = torch.nn.LSTM(
            input_size=8, hidden_size=lstm_hidden_size, num_layers=1
        )

        # fully connected for future duv prediction
        self._fc = torch.nn.Linear(in_features=lstm_hidden_size, out_features=8)  # duv

        self._model = torch.nn.ModuleDict(
            {"lstm": self._lstm, "fc": self._fc}  # forward duv
        )

        self._distance_loss = nn.PairwiseDistance()

        self.lr = lr
        self._betas = betas
        self._val_logged = False

        self._frame_stride = frame_stride

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

    def on_train_epoch_start(self) -> None:
        self._homography_regression = self._homography_regression.eval()
        self._homography_regression.freeze()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, betas=self._betas
        )
        return optimizer

    def forward(self, duvs) -> Any:
        duvs = duvs.permute(1, 0, 2, 3)  # BxTx4x2 -> TxBx4x2
        duvs = duvs.view(duvs.shape[:2] + (-1,))  # TxBx4x2 -> TxBx8
        duvs_pred, (hn, cn) = self._lstm(
            duvs
        )  # duvs_pred gives access to all hidden states in the sequence
        duvs_pred = self._fc(duvs_pred)
        duvs_pred = duvs_pred.view(
            duvs_pred.shape[:2]
            + (
                4,
                2,
            )
        )  # TxBx8 -> TxBx4x2
        duvs_pred = duvs_pred.permute(1, 0, 2, 3)  # TxBx4x2 -> BxTx4x2
        return duvs_pred

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        if self._homography_regression is None:
            raise ValueError("Homography regression model required in training step.")
        videos, transformed_videos, frame_idcs, vid_idcs = batch
        videos, transformed_videos = (
            videos.float() / 255.0,
            transformed_videos.float() / 255.0,
        )

        frames_i, frames_ips = frame_pairs(videos, self._frame_stride)  # re-sort images
        frames_i = frames_i.reshape(
            (-1,) + frames_i.shape[-3:]
        )  # reshape BxNxCxHxW -> B*NxCxHxW
        frames_ips = frames_ips.reshape(
            (-1,) + frames_ips.shape[-3:]
        )  # reshape BxNxCxHxW -> B*NxCxHxW

        with torch.no_grad():
            duvs_reg = self._homography_regression(frames_i, frames_ips)
            duvs_reg = duvs_reg.view(videos.shape[0], -1, 4, 2)  # BxTx4x2

        # forward
        duvs_pred = self(duvs_reg[:, :-1])

        # compute distance loss
        distance_loss = self._distance_loss(
            duvs_pred.reshape(-1, 2),
            duvs_reg[:, 1:].reshape(-1, 2),  # note that the first value is skipped
        ).mean()

        self.log("train/distance", distance_loss)
        return distance_loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        if self._homography_regression is None:
            raise ValueError("Homography regression model required in training step.")
        # by default without grad (torch.set_grad_enabled(False))
        videos, transformed_videos, frame_idcs, vid_idcs = batch
        videos, transformed_videos = (
            videos.float() / 255.0,
            transformed_videos.float() / 255.0,
        )

        frames_i, frames_ips = frame_pairs(videos, self._frame_stride)  # re-sort images
        frames_i = frames_i.reshape(
            (-1,) + frames_i.shape[-3:]
        )  # reshape BxNxCxHxW -> B*NxCxHxW
        frames_ips = frames_ips.reshape(
            (-1,) + frames_ips.shape[-3:]
        )  # reshape BxNxCxHxW -> B*NxCxHxW

        duvs_reg = self._homography_regression(frames_i, frames_ips)
        duvs_reg = duvs_reg.view(videos.shape[0], -1, 4, 2)  # BxTx4x2

        # forward
        duvs_pred = self(duvs_reg[:, :-1])

        # compute distance loss
        distance_loss = self._distance_loss(
            duvs_pred.reshape(-1, 2),
            duvs_reg[:, 1:].reshape(-1, 2),  # note that the first value is skipped
        ).mean()

        # logging
        if not self._val_logged:
            self._val_logged = True
            frames_i = frames_i.view(
                videos.shape[0], -1, 3, videos.shape[-2], videos.shape[-1]
            )  # reshape B*NxCxHxW -> BxNxCxHxW
            frames_ips = frames_ips.view(
                videos.shape[0], -1, 3, videos.shape[-2], videos.shape[-1]
            )  # reshape B*NxCxHxW -> BxNxCxHxW

            # visualize sequence N in zeroth batch
            blends = create_blend_from_four_point_homography(
                frames_i[0], frames_ips[0], duvs_reg[0]
            )

            self.logger.experiment.add_images("val/blend", blends, self.global_step)

            uv = image_edges(frames_i[0, 0].unsqueeze(0))
            uv_reg = integrate_duv(
                uv, duvs_reg[0, 1:]
            )  # batch 0, note that first value is skipped
            uv_pred = integrate_duv(uv, duvs_pred[0])  # batch 0
            uv_traj_fig = uv_trajectory_figure(
                uv_reg.cpu().numpy(), uv_pred.detach().cpu().numpy()
            )
            self.logger.experiment.add_figure(
                "val/uv_traj_fig", uv_traj_fig, self.global_step
            )

        self.log("val/distance", distance_loss)

    def on_validation_epoch_end(self) -> None:
        self._val_logged = False
        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        pass

    def _create_blend_from_homography_regression(
        self, frames_i: torch.Tensor, frames_ips: torch.Tensor, duvs: torch.Tensor
    ):
        r"""Helper function that creates blend figure, given four point homgraphy representation.

        Args:
            frames_i (torch.Tensor): Frames i of shape NxCxHxW
            frames_ips (torch.Tensor): Frames i+step of shape NxCxHxW
            duvs (torch.Tensor): Edge delta from frames i+step to frames i of shape Nx4x2

        Return:
            blends (torch.Tensor): Blends of warp(frames_i) and frames_ips
        """
        uvs = image_edges(frames_i)
        Hs = four_point_homography_to_matrix(uvs, duvs)
        try:  # handle inversion error
            wrps = warp_perspective(frames_i, torch.inverse(Hs), frames_i.shape[-2:])
            blends = yt_alpha_blend(frames_ips, wrps)
        except:
            return frames_i
        return blends


class LSTMModule(pl.LightningModule):
    def __init__(
        self,
        lstm_hidden_size: int = 512,
        lr: float = 1e-4,
        betas: List[float] = [0.9, 0.999],
        frame_stride: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters("lr", "betas")

        # load model
        self._lstm = torch.nn.LSTM(
            input_size=8,
            hidden_size=lstm_hidden_size,
            num_layers=1,
        )

        # fully connected for future duv prediction
        self._fc = torch.nn.Linear(in_features=lstm_hidden_size, out_features=8)  # duv

        self._model = torch.nn.ModuleDict(
            {"lstm": self._lstm, "fc": self._fc}  # forward duv
        )

        self._distance_loss = nn.PairwiseDistance()

        self.lr = lr
        self._betas = betas
        self._val_logged = False

        self._frame_stride = frame_stride

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, betas=self._betas
        )
        return optimizer

    def forward(self, duvs_i) -> Any:
        duvs_i = duvs_i.permute(1, 0, 2, 3)  # BxTx4x2 -> TxBx4x2
        dduvs_i = differentiate_duv(duvs_i, False)
        dduvs_i = dduvs_i.view(dduvs_i.shape[:2] + (-1,))  # (T-1)xBx4x2 -> (T-1)xBx8
        dduvs_ip1, (hn, cn) = self._lstm(
            dduvs_i
        )  # dduvs_p1 gives access to all hidden states in the sequence
        dduvs_ip1 = self._fc(dduvs_ip1)
        dduvs_ip1 = dduvs_ip1.view(
            dduvs_ip1.shape[:2]
            + (
                4,
                2,
            )
        )  # (T-1)xBx8 -> (T-1)xBx4x2
        dduvs_ip1 = dduvs_ip1.permute(1, 0, 2, 3)  # (T-1)xBx4x2 -> Bx(T-1)x4x2
        duvs_i = duvs_i.permute(1, 0, 2, 3)  # TxBx4x2 -> BxTx4x2
        duvs_ip1 = duvs_i[:, 1:]
        duvs_ip2 = duvs_ip1 + dduvs_ip1
        return duvs_ip2

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        duvs_reg, frame_idcs, vid_idcs = batch
        duvs_reg = duvs_reg.float()

        # forward
        duvs_ip2 = self(duvs_reg)  # Bx(T-1)x4x2

        # compute distance loss
        distance_loss = self._distance_loss(
            duvs_ip2[:, :-1].reshape(
                -1, 2
            ),  # we don't have ground truth for the last value in the sequence
            duvs_reg[:, 2:].reshape(
                -1, 2
            ),  # note that the first two values are skipped
        )

        self.log("train/distance", distance_loss.mean())
        return {
            "loss": distance_loss.mean(),
            "per_sequence_loss": distance_loss.detach()
            .view(duvs_ip2.shape[0], -1)
            .mean(axis=-1)
            .cpu()
            .numpy(),
        }

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        img_seq, duvs_reg, frame_idcs, vid_idcs = batch
        img_seq = img_seq.float() / 255.0
        duvs_reg = duvs_reg.float()

        # forward
        duvs_ip2 = self(duvs_reg)  # Bx(T-1)x4x2

        # compute distance loss
        distance_loss = self._distance_loss(
            duvs_ip2[:, :-1].reshape(
                -1, 2
            ),  # we don't have ground truth for the last value in the sequence
            duvs_reg[:, 2:].reshape(
                -1, 2
            ),  # note that the first two values are skipped
        )

        # # logging
        if not self._val_logged:
            self._val_logged = True
            frames_i, frames_ips = frame_pairs(
                img_seq, self._frame_stride
            )  # re-sort images

            # visualize sequence N in zeroth batch
            blends = create_blend_from_four_point_homography(
                frames_i[0], frames_ips[0], duvs_reg[0, :-1]
            )

            self.logger.experiment.add_images("val/blend", blends, self.global_step)

            uv = image_edges(frames_i[0, 0].unsqueeze(0))
            uv_reg = integrate_duv(
                uv, duvs_reg[0, 1:]
            )  # batch 0, note that first value is skipped
            uv_pred = integrate_duv(uv, duvs_ip2[0])  # batch 0
            uv_traj_fig = uv_trajectory_figure(
                uv_reg.cpu().numpy(), uv_pred.detach().cpu().numpy()
            )
            self.logger.experiment.add_figure(
                "val/uv_traj_fig", uv_traj_fig, self.global_step
            )

        self.log("val/distance", distance_loss.mean())

    def on_validation_epoch_end(self) -> None:
        self._val_logged = False
        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        pass


class FeatureLSTMIncrementalModule(pl.LightningModule):
    def __init__(
        self,
        encoder: dict,
        lstm: dict,
        head: List[dict],
        optimizer: dict,
        loss: dict,
        frame_stride: int = 1,
    ) -> None:
        super().__init__()
        self._encoder = getattr(
            importlib.import_module(encoder["module"]), encoder["name"]
        )(**encoder["kwargs"])

        self._lstm = getattr(importlib.import_module(lstm["module"]), lstm["name"])(
            **lstm["kwargs"]
        )

        modules = []
        for module in head:
            modules.append(
                getattr(importlib.import_module(module["module"]), module["name"])(
                    **module["kwargs"]
                )
            )
        self._head = torch.nn.Sequential(*modules)

        self._optimizer = getattr(
            importlib.import_module(optimizer["module"]), optimizer["name"]
        )(params=self.parameters(), **optimizer["kwargs"])

        self._loss = getattr(importlib.import_module(loss["module"]), loss["name"])(
            **loss["kwargs"]
        )

        self._val_logged = False
        self._frame_stride = frame_stride

        self._taylor = TaylorHomographyPrediction(
            order=1
        )  # comparing against simple linear model

    def inject_homography_regression(
        self, homography_regression: dict, homography_regression_prefix: str
    ):
        raise RuntimeError("Currently not supported.")

    def forward(
        self,
        imgs_ip1: torch.Tensor,
        duvs_i: torch.Tensor,
        dduvs_im1: torch.Tensor,
        hx: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # for indices, see https://drive.google.com/file/d/1T1HV01G0bzM_xAavhefhGgsOHGRQmCGw/view?usp=share_link
        B, T, C, H, W = imgs_ip1.shape

        # forward videos into latent space
        imgs_ip1 = imgs_ip1.reshape(-1, C, H, W)  # BxTxCxHxW -> B*TxCxHxW
        f_ip1 = self._encoder(imgs_ip1)
        f_ip1 = f_ip1.view(B, T, -1)  # B*TxF -> BxTxF, where F = features
        f_ip1 = f_ip1.permute(1, 0, 2)  # BxTxF -> TxBxF

        # prepare dduv
        dduvs_im1 = dduvs_im1.view(B, T, -1)  # BxTx4x2 -> BxTx8
        dduvs_im1 = dduvs_im1.permute(1, 0, 2)  # BxTx8 -> TxBx8

        # lstm and head
        f_ip1 = torch.concat([f_ip1, dduvs_im1], axis=-1)
        dduvs_i, hx = self._lstm(f_ip1, hx)
        dduvs_i = self._head(dduvs_i)

        dduvs_i = dduvs_i.view(T, B, 4, 2)  # TxBx8 -> TxBx4x2
        dduvs_i = dduvs_i.permute(1, 0, 2, 3)  # TxBx4x2 -> BxTx4x2
        duvs_ip1 = duvs_i + dduvs_i

        return duvs_ip1, hx

    def configure_optimizers(self) -> Any:
        return self._optimizer

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        (
            tf_imgs,
            duvs_reg,
            frame_idcs,
            vid_idcs,
        ) = batch  # transformed images and four point homography
        tf_imgs = tf_imgs.float() / 255.0
        duvs_reg = duvs_reg.float()

        # forward model
        dduvs_reg = differentiate_duv(duvs_reg, True)  # Bx(T-1)x4x2
        duvs_ip1, _ = self(tf_imgs[:, 2:], duvs_reg[:, 1:-1], dduvs_reg[:, :-1])

        # compute loss
        loss = self._loss(
            duvs_ip1.reshape(-1, 2),
            duvs_reg[:, 2:].reshape(
                -1, 2
            ),  # note that the first two values are skipped
        )

        self.log("train/loss", loss.mean())

        return {
            "loss": loss.mean(),
            "per_sequence_loss": loss.detach()
            .view(duvs_ip1.shape[0], -1)
            .mean(axis=-1)
            .cpu()
            .numpy(),
        }

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        (
            tf_imgs,
            duvs_reg,
            frame_idcs,
            vid_idcs,
        ) = batch  # transformed images and four point homography
        tf_imgs = tf_imgs.float() / 255.0
        duvs_reg = duvs_reg.float()

        # forward model
        dduvs_reg = differentiate_duv(duvs_reg, True)  # Bx(T-1)x4x2
        duvs_ip1, _ = self(tf_imgs[:, 2:], duvs_reg[:, 1:-1], dduvs_reg[:, :-1])

        # compute loss
        loss = self._loss(
            duvs_ip1.reshape(-1, 2),
            duvs_reg[:, 2:].reshape(
                -1, 2
            ),  # note that the first two values are skipped
        )

        self.log("val/loss", loss.mean())

        if not self._val_logged:
            self._val_logged = True
            frames_i, frames_ips = frame_pairs(
                tf_imgs, self._frame_stride
            )  # re-sort images

            # visualize sequence N in zeroth batch
            blends = create_blend_from_four_point_homography(
                frames_i[0], frames_ips[0], duvs_reg[0, :-1]
            )

            self.logger.experiment.add_images("val/blend", blends, self.global_step)

            uv = image_edges(frames_i[0, 0].unsqueeze(0))
            uv_reg = integrate_duv(
                uv, duvs_reg[0, 1:]
            )  # batch 0, note that first value is skipped
            uv_pred = integrate_duv(uv, duvs_ip1[0])  # batch 0
            uv_traj_fig = uv_trajectory_figure(
                uv_reg.cpu().numpy(), uv_pred.detach().cpu().numpy()
            )
            self.logger.experiment.add_figure(
                "val/uv_traj_fig", uv_traj_fig, self.global_step
            )

        # classical estimation
        duvs_ip1_taylor = self._taylor(duvs_reg.cpu())[:, 1:]

        # compute loss
        loss_taylor = self._loss(
            duvs_ip1_taylor.reshape(
                -1, 2
            ),  # we don't have ground truth for the last value in the sequence
            duvs_reg[:, 2:]
            .cpu()
            .reshape(-1, 2),  # note that the first two values are skipped
        )
        self.log("val/loss_taylor", loss_taylor.mean())
        self.log("val/taylor_loss_minus_loss", loss_taylor.mean() - loss.mean())

    def on_validation_epoch_end(self) -> None:
        self._val_logged = False
        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        pass


class FeatureLSTMModule(pl.LightningModule):
    def __init__(
        self,
        encoder: dict,
        lstm: dict,
        head: List[dict],
        optimizer: dict,
        loss: dict,
        frame_stride: int = 1,
    ) -> None:
        super().__init__()
        self._encoder = getattr(
            importlib.import_module(encoder["module"]), encoder["name"]
        )(**encoder["kwargs"])

        self._lstm = getattr(importlib.import_module(lstm["module"]), lstm["name"])(
            **lstm["kwargs"]
        )

        modules = []
        for module in head:
            modules.append(
                getattr(importlib.import_module(module["module"]), module["name"])(
                    **module["kwargs"]
                )
            )
        self._head = torch.nn.Sequential(*modules)

        self._optimizer = getattr(
            importlib.import_module(optimizer["module"]), optimizer["name"]
        )(params=self.parameters(), **optimizer["kwargs"])

        self._loss = getattr(importlib.import_module(loss["module"]), loss["name"])(
            **loss["kwargs"]
        )

        self._sign = 1.
        if isinstance(self._loss, torch.nn.CosineSimilarity):
            self._sign = -1.

        self._val_logged = False
        self._frame_stride = frame_stride

        self._taylor = TaylorHomographyPrediction(
            order=1
        )  # comparing against simple linear model

    def inject_homography_regression(
        self, homography_regression: dict, homography_regression_prefix: str
    ):
        raise RuntimeError("Currently not supported.")

    def forward(
        self,
        imgs: torch.Tensor,
        hx: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C, H, W = imgs.shape

        # forward videos into latent space
        imgs = imgs.reshape(-1, C, H, W)  # BxTxCxHxW -> B*TxCxHxW
        f = self._encoder(imgs)
        f = f.view(B, T, -1)  # B*TxF -> BxTxF, where F = features
        f = f.permute(1, 0, 2)  # BxTxF -> TxBxF

        # lstm and head
        duvs, hx = self._lstm(f, hx)
        duvs = self._head(duvs)

        duvs = duvs.view(T, B, 4, 2)  # TxBx8 -> TxBx4x2
        duvs = duvs.permute(1, 0, 2, 3)  # TxBx4x2 -> BxTx4x2

        return duvs, hx

    def configure_optimizers(self) -> Any:
        return self._optimizer

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        (
            tf_imgs,
            duvs_reg,
            frame_idcs,
            vid_idcs,
        ) = batch  # transformed images and four point homography
        tf_imgs = tf_imgs.float() / 255.0
        duvs_reg = duvs_reg.float()

        # forward model
        duvs, _ = self(tf_imgs)

        # compute loss
        loss = self._sign*self._loss(
            duvs.reshape(-1, 2),
            duvs_reg.reshape(-1, 2),
        )

        self.log("train/loss", loss.mean())

        return {
            "loss": loss.mean(),
            "per_sequence_loss": loss.detach()
            .view(duvs.shape[0], -1)
            .mean(axis=-1)
            .cpu()
            .numpy(),
        }

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        (
            tf_imgs,
            duvs_reg,
            frame_idcs,
            vid_idcs,
        ) = batch  # transformed images and four point homography
        tf_imgs = tf_imgs.float() / 255.0
        duvs_reg = duvs_reg.float()

        # forward model
        duvs, _ = self(tf_imgs)

        # compute loss
        loss = self._sign*self._loss(
            duvs.reshape(-1, 2),
            duvs_reg.reshape(-1, 2),
        )

        self.log("val/loss", loss.mean())

        if not self._val_logged:
            self._val_logged = True
            frames_i, frames_ips = frame_pairs(
                tf_imgs, self._frame_stride
            )  # re-sort images

            # visualize sequence N in zeroth batch
            blends = create_blend_from_four_point_homography(
                frames_i[0], frames_ips[0], duvs_reg[0, :-1]
            )

            self.logger.experiment.add_images("val/blend", blends, self.global_step)

            uv = image_edges(frames_i[0, 0].unsqueeze(0))
            uv_reg = integrate_duv(uv, duvs_reg[0])  # batch 0
            uv_pred = integrate_duv(uv, duvs[0])  # batch 0
            uv_traj_fig = uv_trajectory_figure(
                uv_reg.cpu().numpy(), uv_pred.detach().cpu().numpy()
            )
            self.logger.experiment.add_figure(
                "val/uv_traj_fig", uv_traj_fig, self.global_step
            )

        # classical estimation
        duvs_taylor = self._taylor(duvs_reg.cpu())

        # compute loss
        loss_taylor = self._sign*self._loss(
            duvs_taylor.reshape(
                -1, 2
            ),  # we don't have ground truth for the last value in the sequence
            duvs_reg[:, self._taylor._order :]
            .cpu()
            .reshape(-1, 2),  # note that the first two values are skipped
        )
        self.log("val/loss_taylor", loss_taylor.mean())
        self.log("val/taylor_loss_minus_loss", loss_taylor.mean() - loss.mean())

    def on_validation_epoch_end(self) -> None:
        self._val_logged = False
        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        pass
