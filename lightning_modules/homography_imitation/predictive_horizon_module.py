import os
import pytorchvideo
from pytorchvideo.models import r2plus1d
import torch
import torch.nn as nn
# import torchvision.models as models
import pytorch_lightning as pl
from typing import List
from kornia.geometry import warp_perspective
import lightning_modules
from utils.processing import frame_pairs, image_edges, four_point_homography_to_matrix
from utils.viz import yt_alpha_blend, duv_mean_pairwise_distance_figure



from pytorchvideo.models import r2plus1d, resnet, slowfast, vision_transformers, head




# - create both feature and standard forward model
# - have both forward M time steps, and predict the following N
# - state_buffer, preview horizon
# - M, N


class PredictiveHorizonModule(pl.LightningModule):
    def __init__(self, backbone: dict, lr: float=1e-4, betas: List[float]=[0.9, 0.999], log_n_steps: int=1000, recall_horizon: int=4, preview_horizon: int=4, frame_stride: int=1):
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

        # # modify out layers
        # self._model.fc = nn.Linear(
        #     in_features=self._model.fc.in_features,
        #     out_features=8*preview_horizon
        # )

        self._distance_loss = nn.PairwiseDistance()

        self._lr = lr
        self._betas = betas
        self._validation_step_ct = 0
        self._log_n_steps = log_n_steps

        self._recall_horizon = recall_horizon
        self._preview_horizon = preview_horizon
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
        r"""Forward first images.
        """
        return self._model(img).view(-1, self._preview_horizon, 4, 2)

    def training_step(self, batch, batch_idx):
        if self._homography_regression is None:
            raise ValueError('Homography regression model required in training step.')
        videos, transformed_videos, frame_rate, vid_fps, vid_idc, clip_idc = batch
        frames_i, frames_ips = frame_pairs(videos[:,-(self._preview_horizon+1):], self._frame_stride)  # re-sort images
        frames_i   = frames_i.reshape((-1,) + frames_i.shape[-3:])      # reshape BxNxCxHxW -> B*NxCxHxW
        frames_ips = frames_ips.reshape((-1,) + frames_ips.shape[-3:])  # reshape BxNxCxHxW -> B*NxCxHxW

        with torch.no_grad():
            duvs_reg = self._homography_regression(frames_i, frames_ips)
            duvs_reg = duvs_reg.view(videos.shape[0], -1, 4, 2)  # reshape B*Nx4x2 -> BxNx4x2

            # logging
            if self.global_step % self._log_n_steps == 0:
                frames_i   = frames_i.view(videos.shape[0], -1, 3, videos.shape[-2], videos.shape[-1])   # reshape B*NxCxHxW -> BxNxCxHxW
                frames_ips = frames_ips.view(videos.shape[0], -1, 3, videos.shape[-2], videos.shape[-1]) # reshape B*NxCxHxW -> BxNxCxHxW

                # visualize sequence N in zeroth batch
                blends = self._create_blend_from_homography_regression(frames_i[0], frames_ips[0], duvs_reg[0])

                self.logger.experiment.add_images('verify/blend_train', blends, self.global_step)

                # visualize duv mean pairwise distance to zero
                duv_mpd_seq_figure = duv_mean_pairwise_distance_figure(duvs_reg[0].cpu().numpy(), re_fps=frame_rate[0].item(), fps=vid_fps[vid_idc[0]][0].item())  # get vid_idc of zeroth batch
                self.logger.experiment.add_figure('verify/duv_mean_pairwise_distance', duv_mpd_seq_figure, self.global_step)
        
        recall_horizon = transformed_videos[:,:self._recall_horizon]
        recall_horizon = recall_horizon.permute(0,2,1,3,4)  # BxNxCxHxW -> BxCxNxHxW

        duvs = self(recall_horizon)  # forward recall horizon

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
        frames_i, frames_ips = frame_pairs(videos[:,-(self._preview_horizon+1):], self._frame_stride)  # re-sort images
        frames_i   = frames_i.reshape((-1,) + frames_i.shape[-3:])      # reshape BxNxCxHxW -> B*NxCxHxW
        frames_ips = frames_ips.reshape((-1,) + frames_ips.shape[-3:])  # reshape BxNxCxHxW -> B*NxCxHxW

        duvs_reg = self._homography_regression(frames_i, frames_ips)
        duvs_reg = duvs_reg.view(videos.shape[0], -1, 4, 2)  # reshape B*Nx4x2 -> BxNx4x2

        # only need to sample recall_horizon transformed videos, preview_horizon videos, unless want to forward duv to motion prediction, then sample clip_length_in_frames videos
        recall_horizon = transformed_videos[:,:self._recall_horizon]
        recall_horizon = recall_horizon.permute(0,2,1,3,4)  # BxNxCxHxW -> BxCxNxHxW

        duvs = self(recall_horizon)  # forward recall horizon
        
        # distance loss
        distance_loss = self._distance_loss(
            duvs.view(-1, 2),
            duvs_reg.view(-1, 2)
        ).mean()

        self.log('val/distance', distance_loss, on_epoch=True)
        return distance_loss
        
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