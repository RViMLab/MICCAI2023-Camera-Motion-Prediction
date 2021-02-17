import os
import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from typing import List

import lightning_modules
from utils.processing import framePairs


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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr, betas=self._betas)

    def forward(self, img):
        return self._model(img).view(-1, self._preview_horizon, 4, 2)

    def training_step(self, batch, batch_idx):
        frame_i, frame_ips = framePairs(batch, self._frame_stride)   # re-sort images
        frame_i   = frame_i.reshape((-1,) + frame_i.shape[-3:])      # reshape BxNxCxHxW -> B*NxHxW
        frame_ips = frame_ips.reshape((-1,) + frame_ips.shape[-3:])  # reshape BxNxCxHxW -> B*NxHxW
        with torch.no_grad():
            duv_gt = self._homography_regression(frame_i, frame_ips)

        duv_gt = duv_gt.view(batch.shape[0], -1, 4, 2)  # reshape B*Nx4x2 -> BxNx4x2
        duv = self(batch[:,0].squeeze())  # forward batch of first images
        mse_loss = self._mse_loss(duv, duv_gt)

        return mse_loss

    def validation_step(self, batch, batch_idx):
        frame_i, frame_ips = framePairs(batch, self._frame_stride)   # re-sort images
        frame_i   = frame_i.reshape((-1,) + frame_i.shape[-3:])      # reshape BxNxCxHxW -> B*NxHxW
        frame_ips = frame_ips.reshape((-1,) + frame_ips.shape[-3:])  # reshape BxNxCxHxW -> B*NxHxW
        with torch.no_grad():
            duv_gt = self._homography_regression(frame_i, frame_ips)

            duv_gt = duv_gt.view(batch.shape[0], -1, 4, 2)  # reshape B*Nx4x2 -> BxNx4x2
            duv = self(batch[:,0].squeeze())  # forward batch of first images
            mse_loss = self._mse_loss(duv, duv_gt)

            # # test forwarding after reshape op
            # import cv2
            # from kornia import warp_perspective, tensor_to_image
            # from utils.processing import imageEdges, fourPtToMatrixHomographyRepresentation
            # from utils.viz import yt_alpha_blend
            
            # frame_i   = frame_i.view(batch.shape[0], -1, 3, batch.shape[-2], batch.shape[-1])
            # frame_ips = frame_ips.view(batch.shape[0], -1, 3, batch.shape[-2], batch.shape[-1])

            # for i in range(duv_gt.shape[1]):
            #     uv = imageEdges(frame_i[0])
            #     H = fourPtToMatrixHomographyRepresentation(uv, duv_gt[0])
            #     wrps = warp_perspective(frame_i[0], torch.inverse(H), frame_i[0].shape[-2:])
            #     for j in range(H.shape[0]):
            #         img, wrp = tensor_to_image(frame_ips[0,j]), tensor_to_image(wrps[j])
            #         blend = yt_alpha_blend(img, wrp)
            #         cv2.imshow('blend', blend)
            #         cv2.waitKey()

        return mse_loss
        

    def test_step(self, batch, batch_idx):
        # skip test step until hand labeled homography implemented
        pass
