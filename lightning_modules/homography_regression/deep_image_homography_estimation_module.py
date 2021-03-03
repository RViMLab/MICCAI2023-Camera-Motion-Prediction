import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List
from kornia import tensor_to_image, warp_perspective

from models import DeepHomographyRegression
from utils.viz import warp_figure, yt_alpha_blend
from utils.processing import image_edges, four_point_homography_to_matrix


class DeepImageHomographyEstimationModule(pl.LightningModule):
    def __init__(self, shape: List[int], lr: float=1e-4, betas: List[float]=[0.9, 0.999], log_n_steps: int=1000):
        super().__init__()
        self.save_hyperparameters('lr', 'betas')
        self.model = DeepHomographyRegression(shape)
        self.distance_loss = nn.PairwiseDistance()

        self.lr = lr
        self.betas = betas
        self.validation_step_ct = 0
        self.log_n_steps = log_n_steps

    def forward(self, img, wrp):
        return self.model(img, wrp)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=self.betas)
        return optimizer

    def training_step(self, batch, batch_idx):
        duv_pred = self.model(batch['img_crp'], batch['wrp_crp'])
        distance_loss = self.distance_loss(
            duv_pred.view(-1, 2), 
            batch['duv'].to(duv_pred.dtype).view(-1, 2)
        ).mean()
        self.log('train/distance', distance_loss) # logs all log_every_n_steps https://pytorch-lightning.readthedocs.io/en/latest/logging.html#control-logging-frequency
        return distance_loss    

    def validation_step(self, batch, batch_idx):
        duv_pred = self.model(batch['img_crp'], batch['wrp_crp'])
        distance_loss = self.distance_loss(
            duv_pred.view(-1, 2), 
            batch['duv'].to(duv_pred.dtype).view(-1, 2)
        ).mean()
        self.log('val/distance', distance_loss, on_epoch=True)

        if self.validation_step_ct % self.log_n_steps == 0:
            uv = image_edges(batch['img_crp'])
            H = four_point_homography_to_matrix(uv, duv_pred)
            wrp_pred = warp_perspective(batch['img_crp'], torch.inverse(H), batch['wrp_crp'].shape[-2:])

            blend = yt_alpha_blend(
                batch['wrp_crp'],
                wrp_pred                
            )

            self.logger.experiment.add_images('val/blend', blend, self.validation_step_ct)

            # figure = warp_figure(
            #     img=tensor_to_image(batch['img_pair'][0][0]), 
            #     uv=batch['uv'][0].squeeze().numpy(), 
            #     duv=batch['duv'][0].squeeze().cpu().numpy(), 
            #     duv_pred=duv_pred[0].squeeze().cpu().numpy(), 
            #     H=batch['H'][0].squeeze().numpy()
            # )
            # self.logger.experiment.add_figure('val/wrp', figure, self.validation_step_ct)
        self.validation_step_ct += 1
        return distance_loss

    def test_step(self, batch, batch_idx):
        duv_pred = self.model(batch['img_crp'], batch['wrp_crp'])
        distance_loss = self.distance_loss(
            duv_pred.view(-1, 2), 
            batch['duv'].to(duv_pred.dtype).view(-1, 2)
        ).mean()
        self.log('test/distance', distance_loss, on_epoch=True)
        return distance_loss
