import os
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorchvideo.models import head

import lightning_modules
from utils.processing import frame_pairs, image_edges, integrate_duv
from utils.viz import (create_blend_from_four_point_homography,
                       uv_trajectory_figure)


class PredictiveHorizonModule(pl.LightningModule):
    def __init__(self, backbone: dict, lr: float=1e-4, betas: List[float]=[0.9, 0.999], recall_horizon: int=4, preview_horizon: int=4, frame_stride: int=1):
        super().__init__()
        self.save_hyperparameters('lr', 'betas', 'backbone')

        self._homography_regression = None

        # load model
        self._model = getattr(globals()[backbone['module']], backbone['function']['name'])(**backbone['function']['kwargs'], model_num_class=8*preview_horizon)
        self._model.blocks[-1] = head.ResNetBasicHead(
            pool=self._model.blocks[-1].pool,
            proj=self._model.blocks[-1].proj,
            output_pool=self._model.blocks[-1].output_pool
        )

        self._distance_loss = nn.PairwiseDistance()

        self._lr = lr
        self._betas = betas
        self._val_logged = False

        self._recall_horizon = recall_horizon
        self._preview_horizon = preview_horizon
        self._frame_stride = frame_stride

    def inject_homography_regression(self, homography_regression: dict, homography_regression_prefix: str):
        # load trained homography regression model
        self._homography_regression = getattr(lightning_modules, homography_regression['lightning_module']).load_from_checkpoint(
            checkpoint_path=os.path.join(homography_regression_prefix, homography_regression['path'], homography_regression['checkpoint']),
            **homography_regression['model']
        )
        self._homography_regression = self._homography_regression.eval()
        self._homography_regression.freeze()

    def on_train_epoch_start(self):
        self._homography_regression = self._homography_regression.eval()
        self._homography_regression.freeze()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr, betas=self._betas)
        return optimizer

    def forward(self, img):
        r"""Forward first images.
        """
        return self._model(img).view(-1, self._preview_horizon, 4, 2)

    def training_step(self, batch, batch_idx):
        if self._homography_regression is None:
            raise ValueError('Homography regression model required in training step.')
        # videos, transformed_videos, frame_rate, vid_fps, vid_idc, clip_idc = batch # video dataset
        videos, transformed_videos, frame_idcs, vid_idcs = batch
        videos, transformed_videos = videos.float()/255., transformed_videos.float()/255.

        frames_i, frames_ips = frame_pairs(videos, self._frame_stride)  # re-sort images
        frames_i   = frames_i.reshape((-1,) + frames_i.shape[-3:])      # reshape BxNxCxHxW -> B*NxCxHxW
        frames_ips = frames_ips.reshape((-1,) + frames_ips.shape[-3:])  # reshape BxNxCxHxW -> B*NxCxHxW

        with torch.no_grad():
            duvs_reg = self._homography_regression(frames_i, frames_ips)
            duvs_reg = duvs_reg.view(videos.shape[0], -1, 4, 2)
            duvs_preview_horizon_reg = duvs_reg[:,-self._preview_horizon:]

        recall_horizon = transformed_videos[:,:self._recall_horizon]
        recall_horizon = recall_horizon.permute(0,2,1,3,4)  # BxNxCxHxW -> BxCxNxHxW

        duvs_preview_horizon_pred = self(recall_horizon)  # forward recall horizon

        # distance loss
        distance_loss = self._distance_loss(
            duvs_preview_horizon_pred.view(-1, 2),
            duvs_preview_horizon_reg.reshape(-1, 2)
        ).mean()

        self.log('train/distance', distance_loss)
        return distance_loss

    def validation_step(self, batch, batch_idx):
        if self._homography_regression is None:
            raise ValueError('Homography regression model required in validation step.')
        # by default without grad (torch.set_grad_enabled(False))
        # videos, transformed_videos, frame_rate, vid_fps, vid_idc, clip_idc = batch # video dataset
        videos, transformed_videos, frame_idcs, vid_idcs = batch
        videos, transformed_videos = videos.float()/255., transformed_videos.float()/255.

        frames_i, frames_ips = frame_pairs(videos, self._frame_stride)  # re-sort images
        frames_i   = frames_i.reshape((-1,) + frames_i.shape[-3:])      # reshape BxNxCxHxW -> B*NxCxHxW
        frames_ips = frames_ips.reshape((-1,) + frames_ips.shape[-3:])  # reshape BxNxCxHxW -> B*NxCxHxW

        duvs_reg = self._homography_regression(frames_i, frames_ips)
        duvs_reg = duvs_reg.view(videos.shape[0], -1, 4, 2)  # reshape B*Nx4x2 -> BxNx4x2
        duvs_preview_horizon_reg = duvs_reg[:,-self._preview_horizon:]

        # only need to sample recall_horizon transformed videos, preview_horizon videos, unless want to forward duv to motion prediction, then sample clip_length_in_frames videos
        recall_horizon = transformed_videos[:,:self._recall_horizon]
        recall_horizon = recall_horizon.permute(0,2,1,3,4)  # BxNxCxHxW -> BxCxNxHxW

        duvs_preview_horizon_pred = self(recall_horizon)  # forward recall horizon

        # distance loss
        distance_loss = self._distance_loss(
            duvs_preview_horizon_pred.view(-1, 2),
            duvs_preview_horizon_reg.reshape(-1, 2)
        ).mean()

        # logging
        if not self._val_logged:
            self._val_logged = True
            frames_i   = frames_i.view(videos.shape[0], -1, 3, videos.shape[-2], videos.shape[-1])   # reshape B*NxCxHxW -> BxNxCxHxW
            frames_ips = frames_ips.view(videos.shape[0], -1, 3, videos.shape[-2], videos.shape[-1]) # reshape B*NxCxHxW -> BxNxCxHxW

            # visualize sequence N in zeroth batch
            blends = create_blend_from_four_point_homography(frames_i[0], frames_ips[0], duvs_reg[0])

            self.logger.experiment.add_images('val/blend', blends, self.global_step)

            uv = image_edges(frames_i[0,0].unsqueeze(0))
            uv_reg = integrate_duv(uv, duvs_preview_horizon_reg[0])  # batch 0
            uv_pred = integrate_duv(uv, duvs_preview_horizon_pred[0])  # batch 0
            uv_traj_fig = uv_trajectory_figure(uv_reg.cpu().numpy(), uv_pred.detach().cpu().numpy())
            self.logger.experiment.add_figure('val/uv_traj_fig', uv_traj_fig, self.global_step)

        self.log('val/distance', distance_loss, on_epoch=True)
        
    def on_validation_epoch_end(self) -> None:
        self._val_logged = False
        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        # skip test step until hand labeled homography implemented
        pass
