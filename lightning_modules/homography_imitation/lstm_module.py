import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
from pycls.models import model_zoo
from kornia.geometry import warp_perspective
from typing import List

import lightning_modules
from utils.processing import frame_pairs, image_edges, four_point_homography_to_matrix, integrate_duv
from utils.viz import duv_mean_pairwise_distance_figure, yt_alpha_blend, uv_trajectory_figure


class DuvLSTMModule(pl.LightningModule):
    def __init__(self, lstm_hidden_size: int=512, lr: float=1e-4, betas: List[float]=[0.9, 0.999], log_n_steps: int=1000, frame_stride: int=1) -> None:
        super().__init__()
        self.save_hyperparameters('lr', 'betas')

        self._homography_regression = None

        # load model
        self._lstm = torch.nn.LSTM(
            input_size=8,
            hidden_size=lstm_hidden_size,
            num_layers=1
        )

        # fully connected for future duv prediction
        self._fc = torch.nn.Linear(
            in_features=lstm_hidden_size,
            out_features=8  # duv
        )

        self._model = torch.nn.ModuleDict({
            "lstm": self._lstm,  # forward duv/cumsum(duv) or combination of both
            "fc": self._fc
        })

        self._distance_loss = nn.PairwiseDistance()

        self.lr = lr
        self._betas = betas
        self._validation_step_ct = 0
        self._log_n_steps = log_n_steps

        self._frame_stride = frame_stride

    def inject_homography_regression(self, homography_regression: dict, homography_regression_prefix: str):
        # load trained homography regression model
        self._homography_regression = getattr(lightning_modules, homography_regression['lightning_module']).load_from_checkpoint(
            checkpoint_path=os.path.join(homography_regression_prefix, homography_regression['path'], homography_regression['checkpoint']),
            **homography_regression['model']
        )
        self._homography_regression = self._homography_regression.eval()
        self._homography_regression.freeze()

    def on_train_epoch_start(self) -> None:
        self._homography_regression = self._homography_regression.eval()
        self._homography_regression.freeze()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, betas=self._betas)
        return optimizer

    def forward(self, duvs):
        duvs = duvs.permute(1,0,2,3) # BxTx4x2 -> TxBx4x2
        duvs = duvs.view(duvs.shape[:2] + (-1,)) # TxBx4x2 -> TxBx8
        duvs_pred, (hn, cn) = self._lstm(duvs) # duvs_pred gives access to all hidden states in the sequence
        duvs_pred = self._fc(duvs_pred)
        duvs_pred = duvs_pred.view(duvs_pred.shape[:2] + (4, 2,)) # TxBx8 -> TxBx4x2
        duvs_pred = duvs_pred.permute(1,0,2,3) # TxBx4x2 -> BxTx4x2
        return duvs_pred
    
    def training_step(self, batch, batch_idx):
        if self._homography_regression is None:
            raise ValueError('Homography regression model required in training step.')
        videos, transformed_videos, frame_idcs, vid_idcs = batch
        videos, transformed_videos = videos.float()/255., transformed_videos.float()/255.

        frames_i, frames_ips = frame_pairs(videos, self._frame_stride)  # re-sort images
        frames_i   = frames_i.reshape((-1,) + frames_i.shape[-3:])      # reshape BxNxCxHxW -> B*NxCxHxW
        frames_ips = frames_ips.reshape((-1,) + frames_ips.shape[-3:])  # reshape BxNxCxHxW -> B*NxCxHxW

        with torch.no_grad():
            duvs_reg = self._homography_regression(frames_i, frames_ips)
            duvs_reg = duvs_reg.view(videos.shape[0], -1, 4, 2) # BxTx4x2

        # forward
        duvs_pred = self(duvs_reg[:,:-1])

        # compute distance loss
        distance_loss = self._distance_loss(
            duvs_pred.reshape(-1, 2),
            duvs_reg[:,1:].reshape(-1, 2) # note that the first value is skipped
        ).mean()

        # cum_distance_loss = self._distance_loss(
        #     torch.cumsum(duvs_pred, dim=1).reshape(-1, 2),
        #     torch.cumsum(duvs_reg[:,1:], dim=1).reshape(-1, 2)
        # ).mean()

        # logging
        if self.global_step % self._log_n_steps == 0:
            frames_i   = frames_i.view(videos.shape[0], -1, 3, videos.shape[-2], videos.shape[-1])   # reshape B*NxCxHxW -> BxNxCxHxW
            frames_ips = frames_ips.view(videos.shape[0], -1, 3, videos.shape[-2], videos.shape[-1]) # reshape B*NxCxHxW -> BxNxCxHxW

            # visualize sequence N in zeroth batch
            blends = self._create_blend_from_homography_regression(frames_i[0], frames_ips[0], duvs_reg[0])

            self.logger.experiment.add_images('train/blend_train', blends, self.global_step)

            uv = image_edges(frames_i[0,0].unsqueeze(0))
            uv_reg = integrate_duv(uv, duvs_reg[0,1:])  # batch 0, note that first value is skipped
            uv_pred = integrate_duv(uv, duvs_pred[0])  # batch 0
            uv_traj_fig = uv_trajectory_figure(uv_reg.cpu().numpy(), uv_pred.detach().cpu().numpy())
            self.logger.experiment.add_figure('train/uv_traj_fig', uv_traj_fig, self.global_step)

        self.log('train/distance', distance_loss)
        # self.log('train/cum_distance', cum_distance_loss)
        return distance_loss # + cum_distance_loss

    def validation_step(self, batch, batch_idx):
        if self._homography_regression is None:
            raise ValueError('Homography regression model required in training step.')
        # by default without grad (torch.set_grad_enabled(False))
        videos, transformed_videos, frame_idcs, vid_idcs = batch
        videos, transformed_videos = videos.float()/255., transformed_videos.float()/255.

        frames_i, frames_ips = frame_pairs(videos, self._frame_stride)  # re-sort images
        frames_i   = frames_i.reshape((-1,) + frames_i.shape[-3:])      # reshape BxNxCxHxW -> B*NxCxHxW
        frames_ips = frames_ips.reshape((-1,) + frames_ips.shape[-3:])  # reshape BxNxCxHxW -> B*NxCxHxW

        duvs_reg = self._homography_regression(frames_i, frames_ips)
        duvs_reg = duvs_reg.view(videos.shape[0], -1, 4, 2) # BxTx4x2

        # forward
        duvs_pred = self(duvs_reg[:,:-1])

        # compute distance loss
        distance_loss = self._distance_loss(
            duvs_pred.reshape(-1, 2),
            duvs_reg[:,1:].reshape(-1, 2) # note that the first value is skipped
        ).mean()

        # cum_distance_loss = self._distance_loss(
        #     torch.cumsum(duvs_pred, dim=1).reshape(-1, 2),
        #     torch.cumsum(duvs_reg[:,1:], dim=1).reshape(-1, 2)
        # ).mean()

        # logging
        if self._validation_step_ct % self._log_n_steps == 0:
            frames_i   = frames_i.view(videos.shape[0], -1, 3, videos.shape[-2], videos.shape[-1])   # reshape B*NxCxHxW -> BxNxCxHxW
            frames_ips = frames_ips.view(videos.shape[0], -1, 3, videos.shape[-2], videos.shape[-1]) # reshape B*NxCxHxW -> BxNxCxHxW

            # visualize sequence N in zeroth batch
            blends = self._create_blend_from_homography_regression(frames_i[0], frames_ips[0], duvs_reg[0])

            self.logger.experiment.add_images('val/blend_train', blends, self._validation_step_ct)

            uv = image_edges(frames_i[0,0].unsqueeze(0))
            uv_reg = integrate_duv(uv, duvs_reg[0,1:])  # batch 0, note that first value is skipped
            uv_pred = integrate_duv(uv, duvs_pred[0])  # batch 0
            uv_traj_fig = uv_trajectory_figure(uv_reg.cpu().numpy(), uv_pred.detach().cpu().numpy())
            self.logger.experiment.add_figure('val/uv_traj_fig', uv_traj_fig, self._validation_step_ct)

        self.log('val/distance', distance_loss)
        # self.log('val/cum_distance', cum_distance_loss)
        self._validation_step_ct += 1

    def test_step(self, batch, batch_idx):
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
        try:  # handle inversion error
            wrps = warp_perspective(frames_i, torch.inverse(Hs), frames_i.shape[-2:])
            blends = yt_alpha_blend(frames_ips, wrps)
        except:
            return frames_i
        return blends


class LSTMModule(pl.LightningModule):
    def __init__(self, lstm_hidden_size: int=512, lr: float=1e-4, betas: List[float]=[0.9, 0.999], log_n_steps: int=1000, frame_stride: int=1) -> None:
        super().__init__()
        self.save_hyperparameters('lr', 'betas')

        # load model
        self._lstm = torch.nn.LSTM(
            input_size=8,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            dropout=0.5
        )

        # fully connected for future duv prediction
        self._fc = torch.nn.Linear(
            in_features=lstm_hidden_size,
            out_features=8  # duv
        )

        self._model = torch.nn.ModuleDict({
            "lstm": self._lstm,  # forward duv/cumsum(duv) or combination of both
            "fc": self._fc
        })

        self._distance_loss = nn.PairwiseDistance()

        self.lr = lr
        self._betas = betas
        self._validation_step_ct = 0
        self._log_n_steps = log_n_steps

        self._frame_stride = frame_stride

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, betas=self._betas)
        return optimizer

    def forward(self, duvs):
        duvs = duvs.permute(1,0,2,3) # BxTx4x2 -> TxBx4x2
        duvs = duvs.view(duvs.shape[:2] + (-1,)) # TxBx4x2 -> TxBx8
        duvs_pred, (hn, cn) = self._lstm(duvs) # duvs_pred gives access to all hidden states in the sequence
        duvs_pred = self._fc(duvs_pred)
        duvs_pred = duvs_pred.view(duvs_pred.shape[:2] + (4, 2,)) # TxBx8 -> TxBx4x2
        duvs_pred = duvs_pred.permute(1,0,2,3) # TxBx4x2 -> BxTx4x2
        return duvs_pred
    
    def training_step(self, batch, batch_idx):
        duvs_reg, frame_idcs, vid_idcs = batch
        duvs_reg = duvs_reg.float()

        # forward
        duvs_pred = self(duvs_reg[:,:-1])

        # compute distance loss
        distance_loss = self._distance_loss(
            duvs_pred.reshape(-1, 2),
            duvs_reg[:,1:].reshape(-1, 2) # note that the first value is skipped
        )

        self.log('train/distance', distance_loss.mean())
        return {
            'loss': distance_loss.mean(),
            'sequence_loss': distance_loss.detach().view(duvs_pred.shape[:2] + (4,)).mean(axis=-1).cpu().numpy(), 
            'frame_idcs': frame_idcs.cpu().numpy(),
            'vid_idcs': vid_idcs.cpu().numpy()
        }

    def validation_step(self, batch, batch_idx):
        img_seq, duvs_reg, frame_idcs, vid_idcs = batch
        img_seq = img_seq.float()/255.
        duvs_reg = duvs_reg.float()

        # forward
        duvs_pred = self(duvs_reg[:,:-1])

        # compute distance loss
        distance_loss = self._distance_loss(
            duvs_pred.reshape(-1, 2),
            duvs_reg[:,1:].reshape(-1, 2) # note that the first value is skipped
        )

        # # logging
        if self._validation_step_ct % self._log_n_steps == 0:
            frames_i, frames_ips = frame_pairs(img_seq, self._frame_stride)  # re-sort images

            # visualize sequence N in zeroth batch
            blends = self._create_blend_from_homography_regression(frames_i[0], frames_ips[0], duvs_reg[0,:-1])

            self.logger.experiment.add_images('val/blend_train', blends, self._validation_step_ct)

            uv = image_edges(frames_i[0,0].unsqueeze(0))
            uv_reg = integrate_duv(uv, duvs_reg[0,1:])  # batch 0, note that first value is skipped
            uv_pred = integrate_duv(uv, duvs_pred[0])  # batch 0
            uv_traj_fig = uv_trajectory_figure(uv_reg.cpu().numpy(), uv_pred.detach().cpu().numpy())
            self.logger.experiment.add_figure('val/uv_traj_fig', uv_traj_fig, self._validation_step_ct)

        self.log('val/distance', distance_loss.mean())
        self._validation_step_ct += 1

    def test_step(self, batch, batch_idx):
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
        try:  # handle inversion error
            wrps = warp_perspective(frames_i, torch.inverse(Hs), frames_i.shape[-2:])
            blends = yt_alpha_blend(frames_ips, wrps)
        except:
            return frames_i
        return blends


class FeatureLSTMModule(pl.LightningModule):
    def __init__(self, backbone: dict, head: dict, lr: float=1e-4, betas: List[float]=[0.9, 0.999], log_n_steps: int=1000, frame_stride: int=1):
        super().__init__()
        self.save_hyperparameters('lr', 'betas', 'backbone')
        backbone_dict = {
            'ResNet-18': 'resnet18',
            'ResNet-34': 'resnet34'
        }
        if backbone['name'] == 'ResNet-18' or backbone['name'] == 'ResNet-34':
            backbone['name'] = backbone_dict[backbone['name']]

        if backbone['name'] == 'resnet18' or backbone['name'] == 'resnet34':
            self._encoder = getattr(models, backbone['name'])(**backbone['kwargs'])

            self._encoder.fc = torch.nn.Linear(
                in_features=self._encoder.fc.in_features,
                out_features=backbone['backbone_features']
            )
        else:
            if backbone['name'] not in model_zoo.get_model_list():
                raise ValueError('Model {} not available.'.format(backbone['name']))
            self._encoder = model_zoo.build_model(backbone['name'], **backbone['kwargs'])

            self._encoder.head.fc = torch.nn.Linear(
                in_features=self._encoder.head.fc.in_features,
                out_features=backbone['backbone_features']
            )

        # lstm on encoded features
        head['kwargs']['input_size'] = backbone['backbone_features'] + 8
        self._lstm = torch.nn.LSTM(**head['kwargs'])

        # fully connected for future duv prediction
        self._fc = torch.nn.Linear(
            in_features=head['kwargs']['hidden_size'],
            out_features=8  # duv
        )

        self._homography_regression = None

        self._distance_loss = nn.PairwiseDistance()

        self.lr = lr  # naming required by lightning auto-lr-finder https://pytorch-lightning.readthedocs.io/en/latest/advanced/lr_finder.html#using-lightning-s-built-in-lr-finder
        self._betas = betas
        self._validation_step_ct = 0
        self._log_n_steps = log_n_steps

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
        optimizer = torch.optim.Adam(self._encoder.parameters(), lr=self.lr, betas=self._betas)
        return optimizer

    def forward(self, imgs: torch.Tensor, duvs: torch.Tensor) -> torch.Tensor:
        r"""Forward past images and past motion to predict future motion.

        Args:
            imgs (torch.Tensor): Images of shape BxTxCxHxW, where T is the sequence length
            duvs (torch.Tensor): Four point homographies of shape BxTx4x2 
        """
        features = self._encoder(imgs.reshape((-1,) + imgs.shape[-3:])) # B*(T-1)xCxHxW, where N == sequence -> returns B*(T-1)x backbone_features (512)
        features = features.reshape(duvs.shape[:2] + (-1,))
        features = features.permute(1,0,2) # BxTxbackbone_features -> TxBxbackbone_features

        # concatenate
        duvs = duvs.permute(1,0,2,3) # BxTx4x2 -> TxBx4x2
        duvs = duvs.view(duvs.shape[:2] + (-1,)) # TxBx4x2 -> TxBx8
        features = torch.concat([features, duvs], dim=-1)

        # forward lstm head
        duvs_pred, (hn, cn) = self._lstm(features) # duvs_pred gives access to all hidden states in the sequence
        duvs_pred = self._fc(duvs_pred)
        duvs_pred = duvs_pred.view(duvs_pred.shape[:2] + (4, 2,)) # TxBx8 -> TxBx4x2
        duvs_pred = duvs_pred.permute(1,0,2,3) # TxBx4x2 -> BxTx4x2
        return duvs_pred

    def training_step(self, batch, batch_idx):
        if self._homography_regression is None:
            raise ValueError('Homography regression model required in training step.')
        # by default without grad (torch.set_grad_enabled(False))
        videos, transformed_videos, frame_idcs, vid_idcs = batch
        videos, transformed_videos = videos.float()/255., transformed_videos.float()/255.

        frames_i, frames_ips = frame_pairs(videos, self._frame_stride)  # re-sort images
        frames_i   = frames_i.reshape((-1,) + frames_i.shape[-3:])      # reshape BxNxCxHxW -> B*NxCxHxW
        frames_ips = frames_ips.reshape((-1,) + frames_ips.shape[-3:])  # reshape BxNxCxHxW -> B*NxCxHxW

        with torch.no_grad():
            duvs_reg = self._homography_regression(frames_i, frames_ips)
            duvs_reg = duvs_reg.view(videos.shape[0], -1, 4, 2) # BxTx4x2

        # forward
        duvs_pred = self(frames_i.reshape(videos.shape[0], -1, 3, videos.shape[-2], videos.shape[-1])[:,:-1], duvs_reg[:,:-1])  # note that last duv value is skipped

        # compute distance loss
        distance_loss = self._distance_loss(
            duvs_pred.reshape(-1, 2),
            duvs_reg[:,1:].reshape(-1, 2) # note that the first value is skipped
        ).mean()

        # # logging
        # if self.global_step % self._log_n_steps == 0:
        #     frames_i   = frames_i.view(videos.shape[0], -1, 3, videos.shape[-2], videos.shape[-1])   # reshape B*NxCxHxW -> BxNxCxHxW
        #     frames_ips = frames_ips.view(videos.shape[0], -1, 3, videos.shape[-2], videos.shape[-1]) # reshape B*NxCxHxW -> BxNxCxHxW

        #     # visualize sequence N in zeroth batch
        #     blends = self._create_blend_from_homography_regression(frames_i[0], frames_ips[0], duvs_reg[0])

        #     self.logger.experiment.add_images('train/blend_train', blends, self.global_step)

        #     uv = image_edges(frames_i[0,0].unsqueeze(0))
        #     uv_reg = integrate_duv(uv, duvs_reg[0,1:])  # batch 0, note that first value is skipped
        #     uv_pred = integrate_duv(uv, duvs_pred[0])  # batch 0
        #     uv_traj_fig = uv_trajectory_figure(uv_reg.cpu().numpy(), uv_pred.detach().cpu().numpy())
        #     self.logger.experiment.add_figure('train/uv_traj_fig', uv_traj_fig, self.global_step)

        self.log('train/distance', distance_loss)
        return distance_loss

    def validation_step(self, batch, batch_idx):
        if self._homography_regression is None:
            raise ValueError('Homography regression model required in training step.')
        # by default without grad (torch.set_grad_enabled(False))
        videos, transformed_videos, frame_idcs, vid_idcs = batch
        videos, transformed_videos = videos.float()/255., transformed_videos.float()/255.

        frames_i, frames_ips = frame_pairs(videos, self._frame_stride)  # re-sort images
        frames_i   = frames_i.reshape((-1,) + frames_i.shape[-3:])      # reshape BxNxCxHxW -> B*NxCxHxW
        frames_ips = frames_ips.reshape((-1,) + frames_ips.shape[-3:])  # reshape BxNxCxHxW -> B*NxCxHxW

        duvs_reg = self._homography_regression(frames_i, frames_ips)
        duvs_reg = duvs_reg.view(videos.shape[0], -1, 4, 2) # BxTx4x2

        # forward
        duvs_pred = self(frames_i.reshape(videos.shape[0], -1, 3, videos.shape[-2], videos.shape[-1])[:,:-1], duvs_reg[:,:-1])  # note that last duv value is skipped

        # compute distance loss
        distance_loss = self._distance_loss(
            duvs_pred.reshape(-1, 2),
            duvs_reg[:,1:].reshape(-1, 2) # note that the first value is skipped
        ).mean()

        # logging
        if self._validation_step_ct % self._log_n_steps == 0:
            frames_i   = frames_i.view(videos.shape[0], -1, 3, videos.shape[-2], videos.shape[-1])   # reshape B*NxCxHxW -> BxNxCxHxW
            frames_ips = frames_ips.view(videos.shape[0], -1, 3, videos.shape[-2], videos.shape[-1]) # reshape B*NxCxHxW -> BxNxCxHxW

            # visualize sequence N in zeroth batch
            blends = self._create_blend_from_homography_regression(frames_i[0], frames_ips[0], duvs_reg[0])

            self.logger.experiment.add_images('val/blend_train', blends, self._validation_step_ct)

            uv = image_edges(frames_i[0,0].unsqueeze(0))
            uv_reg = integrate_duv(uv, duvs_reg[0,1:])  # batch 0, note that first value is skipped
            uv_pred = integrate_duv(uv, duvs_pred[0])  # batch 0
            uv_traj_fig = uv_trajectory_figure(uv_reg.cpu().numpy(), uv_pred.detach().cpu().numpy())
            self.logger.experiment.add_figure('val/uv_traj_fig', uv_traj_fig, self._validation_step_ct)

        self.log('val/distance', distance_loss)
        self._validation_step_ct += 1

    def test_step(self):
        # build a test set first
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
        try:  # handle inversion error
            wrps = warp_perspective(frames_i, torch.inverse(Hs), frames_i.shape[-2:])
            blends = yt_alpha_blend(frames_ips, wrps)
        except:
            return frames_i
        return blends
