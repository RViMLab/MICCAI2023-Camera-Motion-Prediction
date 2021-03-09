import os
import pytorch_lightning as pl
from typing import List

import lightning_modules


class FeatureLSTMModule(pl.LightningModule):
    def __init__(self, shape: List[int], lr: float=1e-4, betas: List[float]=[0.9, 0.999], log_n_steps: int=1000, backbone: str='resnet34', frame_stride: int=1):
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
        self._validation_step_ct = 0
        self._log_n_steps = log_n_steps

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


    def forward(self):
        pass

    def training_step(self):
        pass

    def validation_step(self):
        pass

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
        wrps = warp_perspective(frames_i, torch.inverse(Hs), frames_i.shape[-2:])
        blends = yt_alpha_blend(frames_ips, wrps)
        return blends
