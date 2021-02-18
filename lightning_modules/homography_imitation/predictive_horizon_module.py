import os
import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from typing import List
from kornia import warp_perspective

import lightning_modules
from utils.processing import frame_pairs, image_edges, four_point_homography_to_matrix
from utils.viz import yt_alpha_blend


class PredictiveHorizonModule(pl.LightningModule):
    def __init__(self, shape: List[int], homography_regression: dict, homography_regression_prefix: str, lr: float=1e-4, betas: List[float]=[0.9, 0.999], log_n_steps: int=1000, backbone: str='resnet34', preview_horizon: int=4, frame_stride: int=5):
        super().__init__()
        self.save_hyperparameters('lr', 'betas', 'backbone')

        # load trained homography regression model
        self._homography_regression = getattr(lightning_modules, homography_regression['lightning_module']).load_from_checkpoint(
            checkpoint_path=os.path.join(homography_regression_prefix, homography_regression['path'], homography_regression['checkpoint']),
            **homography_regression['model']
        )
        self._homography_regression.eval()

        # load model
        self._model = getattr(models, backbone)(**{'pretrained': False})

        # modify out layers
        self._model.fc = nn.Linear(
            in_features=self._model.fc.in_features,
            out_features=8*preview_horizon
        )

        self._mse_loss = nn.MSELoss()

        self._lr = lr
        self._betas = betas
        self._validation_step_ct = 0
        self._log_n_steps = log_n_steps

        self._preview_horizon = preview_horizon
        self._frame_stride = frame_stride

    def on_train_start(self):
        self._homography_regression.eval()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr, betas=self._betas)

    def forward(self, img):
        r"""Forward first images, alternatively, forward frames_i to predict duvs to frames_ips
        """
        return self._model(img).view(-1, self._preview_horizon, 4, 2)

    def training_step(self, batch, batch_idx):
        batch = batch.float()/255.
        frames_i, frames_ips = frame_pairs(batch, self._frame_stride)   # re-sort images
        frames_i   = frames_i.reshape((-1,) + frames_i.shape[-3:])      # reshape BxNxCxHxW -> B*NxHxW
        frames_ips = frames_ips.reshape((-1,) + frames_ips.shape[-3:])  # reshape BxNxCxHxW -> B*NxHxW

        torch.set_grad_enabled(False)
        duvs_reg = self._homography_regression(frames_i, frames_ips)
        
        duvs_reg = duvs_reg.view(batch.shape[0], -1, 4, 2)  # reshape B*Nx4x2 -> BxNx4x2

        # VIZ
        if self.global_step % self._log_n_steps == 0:
            frames_i   = frames_i.view(batch.shape[0], -1, 3, batch.shape[-2], batch.shape[-1])
            frames_ips = frames_ips.view(batch.shape[0], -1, 3, batch.shape[-2], batch.shape[-1])

            # visualize zeroth batch
            blends = self._create_blend_from_homography_regression(frames_i[0], frames_ips[0], duvs_reg[0])

            self.logger.experiment.add_images('verify/blend_train', blends, self.global_step)
        # VIZ   

        torch.set_grad_enabled(True)



  
        # duvs_reg = duvs_reg.view(batch.shape[0], -1, 4, 2)  # reshape B*Nx4x2 -> BxNx4x2
        # duv = self(batch[:,0].squeeze())                # forward batch of first images
        # mse_loss = self._mse_loss(duv, duvs_reg)

        # self.log('train/mse', mse_loss)
        # return 0

    def validation_step(self, batch, batch_idx):
        # by default without grad (torch.set_grad_enabled(False))
        batch = batch.float()/255.
        frames_i, frames_ips = frame_pairs(batch, self._frame_stride)   # re-sort images
        frames_i   = frames_i.reshape((-1,) + frames_i.shape[-3:])      # reshape BxNxCxHxW -> B*NxHxW
        frames_ips = frames_ips.reshape((-1,) + frames_ips.shape[-3:])  # reshape BxNxCxHxW -> B*NxHxW

        duvs_reg = self._homography_regression(frames_i, frames_ips)
        duvs_reg = duvs_reg.view(batch.shape[0], -1, 4, 2)  # reshape B*Nx4x2 -> BxNx4x2
        # duv = self(batch[:,0].squeeze())                # forward batch of first images
        # mse_loss = self._mse_loss(duv, duvs_reg)

        # # VIZ
        # frames_i   = frames_i.view(batch.shape[0], -1, 3, batch.shape[-2], batch.shape[-1])
        # frames_ips = frames_ips.view(batch.shape[0], -1, 3, batch.shape[-2], batch.shape[-1])

        # # visualize zeroth batch
        # blends = self._create_blend_from_homography_regression(frames_i[0], frames_ips[0], duvs_reg[0])
        # self.logger.experiment.add_images('verify/blend_val', blends)
        # return mse_loss
        
    def test_step(self, batch, batch_idx):
        # skip test step until hand labeled homography implemented
        pass

    def _create_blend_from_homography_regression(self, frames_i: torch.Tensor, frames_ips: torch.Tensor, duvs: torch.Tensor):
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
        wrps = warp_perspective(frames_i, torch.inverse(Hs), frames_i.shape[-2:])
        blends = yt_alpha_blend(frames_ips, wrps)
        return blends