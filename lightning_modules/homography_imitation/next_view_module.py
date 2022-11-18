import os
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models

import lightning_modules
from utils.processing import frame_pairs
from utils.viz import (create_blend_from_four_point_homography,
                       duv_mean_pairwise_distance_figure)


class NextViewModule(pl.LightningModule):
    def __init__(self, shape: List[int], lr: float=1e-4, betas: List[float]=[0.9, 0.999], backbone: str='resnet34', frame_stride: int=1):
        super().__init__()
        self.save_hyperparameters('lr', 'betas', 'backbone')

        self._homography_regression = None

        # load model
        self._model = getattr(models, backbone)(**{'pretrained': False})

        # modify out layers
        self._model.fc = nn.Linear(
            in_features=self._model.fc.in_features,
            out_features=8
        )

        self._distance_loss = nn.PairwiseDistance()

        self._lr = lr
        self._betas = betas
        self._val_logged = False

        self._frame_stride = frame_stride

    def inject_homography_regression(self, homography_regression: dict, homography_regression_prefix: str):
        # load trained homography regression model
        self._homography_regression = getattr(lightning_modules, homography_regression['lightning_module']).load_from_checkpoint(
            checkpoint_path=os.path.join(homography_regression_prefix, homography_regression['path'], homography_regression['checkpoint']),
            **homography_regression['model']
        )
        self._homography_regression.eval()

    def on_train_epoch_start(self):
        self._homography_regression.eval()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr, betas=self._betas)
        return optimizer

    def forward(self, img):
        r"""Forward frames_i to predict duvs to frames_ips
        """
        return self._model(img).view(-1, 4, 2)

    def training_step(self, batch, batch_idx):
        if self._homography_regression is None:
            raise ValueError('Homography regression model required in training step.')
        videos, transformed_videos, frame_rate, vid_fps, vid_idc, clip_idc = batch

        # homography regression
        frames_i, frames_ips = frame_pairs(videos, self._frame_stride)  # re-sort images
        frames_i   = frames_i.reshape((-1,) + frames_i.shape[-3:])      # reshape BxNxCxHxW -> B*NxHxW
        frames_ips = frames_ips.reshape((-1,) + frames_ips.shape[-3:])  # reshape BxNxCxHxW -> B*NxHxW

        with torch.no_grad():
            duvs_reg = self._homography_regression(frames_i, frames_ips)
            duvs_reg = duvs_reg.view(videos.shape[0], -1, 4, 2)  # reshape B*Nx4x2 -> BxNx4x2

        # homography prediction
        transformed_frames_i = transformed_videos[:,:-self._frame_stride:self._frame_stride]
        transformed_frames_i = transformed_frames_i.reshape((-1,) + frames_i.shape[-3:])  # reshape BxNxCxHxW -> B*NxHxW

        duvs = self(transformed_frames_i)  # forward transformed correspondence to frames_i
        duvs = duvs.view(videos.shape[0], -1, 4, 2)  # reshape B*Nx4x2 -> BxNx4x2

        # distance loss
        distance_loss = self._distance_loss(
            duvs.view(-1, 2),
            duvs_reg.view(-1, 2)
        ).mean()

        self.log('train/distance', distance_loss)
        return distance_loss

    def validation_step(self, batch, batch_idx):
        if self._homography_regression is None:
            raise ValueError('Homography regression model required in validation step.')
        # by default without grad (torch.set_grad_enabled(False))
        videos, transformed_videos, frame_rate, vid_fps, vid_idc, clip_idc = batch

        # homography regression
        frames_i, frames_ips = frame_pairs(videos, self._frame_stride)  # re-sort images
        frames_i   = frames_i.reshape((-1,) + frames_i.shape[-3:])      # reshape BxNxCxHxW -> B*NxHxW
        frames_ips = frames_ips.reshape((-1,) + frames_ips.shape[-3:])  # reshape BxNxCxHxW -> B*NxHxW

        duvs_reg = self._homography_regression(frames_i, frames_ips)
        duvs_reg = duvs_reg.view(videos.shape[0], -1, 4, 2)  # reshape B*Nx4x2 -> BxNx4x2

        # homography prediction
        transformed_frames_i = transformed_videos[:,:-self._frame_stride:self._frame_stride]
        transformed_frames_i = transformed_frames_i.reshape((-1,) + frames_i.shape[-3:])  # reshape BxNxCxHxW -> B*NxHxW

        duvs = self(transformed_frames_i)  # forward transformed correspondence to frames_i
        duvs = duvs.view(videos.shape[0], -1, 4, 2)  # reshape B*Nx4x2 -> BxNx4x2

        if not self._val_logged:
            self._val_logged = True
            # logging
            frames_i   = frames_i.view(videos.shape[0], -1, 3, videos.shape[-2], videos.shape[-1])   # reshape B*NxCxHxW -> BxNxCxHxW
            frames_ips = frames_ips.view(videos.shape[0], -1, 3, videos.shape[-2], videos.shape[-1]) # reshape B*NxCxHxW -> BxNxCxHxW

            # visualize sequence N in zeroth batch
            blends = create_blend_from_four_point_homography(frames_i[0], frames_ips[0], duvs_reg[0])

            self.logger.experiment.add_images('val/blend_train', blends, self.global_step)

            # visualize duv mean pairwise distance to zero
            duv_mpd_seq_figure = duv_mean_pairwise_distance_figure(duvs_reg[0].cpu().numpy(), re_fps=frame_rate[0].item(), fps=vid_fps[vid_idc[0]][0].item())  # get vid_idc of zeroth batch
            self.logger.experiment.add_figure('val/duv_mean_pairwise_distance', duv_mpd_seq_figure, self.global_step)

        # distance loss
        distance_loss = self._distance_loss(
            duvs.view(-1, 2),
            duvs_reg.view(-1, 2)
        ).mean()

        self.log('val/distance', distance_loss, on_epoch=True)
        return distance_loss

    def on_validation_epoch_end(self) -> None:
        self._val_logged = True
        return super().on_validation_epoch_end()
        
    def test_step(self, batch, batch_idx):
        # skip test step until hand labeled homography implemented, therefore, analyze homography histograms
        pass
