import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List

from models import DeepHomographyRegression
from utils.viz import warp_figure


class SupervisedHomographyModule(pl.LightningModule):
    def __init__(self, shape, lr: float=1e-4, betas: List[float]=[0.9, 0.999]):
        super().__init__()
        self.save_hyperparameters('lr', 'betas')
        self.model = DeepHomographyRegression(shape)
        self.loss = nn.PairwiseDistance()

        self.lr = lr
        self.betas = betas

    def forward(self, img, wrp):
        return self.model(img, wrp)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=self.betas)
        return optimizer

    def training_step(self, batch, batch_idx):
        duv_pred = self.model(batch['img_seq'][0], batch['img_seq'][1])
        loss = self.loss(
            duv_pred.view(-1, 2), 
            batch['duv'].to(duv_pred.dtype).view(-1, 2)
        ).mean()
        self.log('train_loss', loss) # logs all log_every_n_steps https://pytorch-lightning.readthedocs.io/en/latest/logging.html#control-logging-frequency
        return loss     

    def validation_step(self, batch, batch_idx):
        duv_pred = self.model(batch['img_seq'][0], batch['img_seq'][1])
        loss = self.loss(
            duv_pred.view(-1, 2), 
            batch['duv'].to(duv_pred.dtype).view(-1, 2)
        ).mean()
        self.log('val_loss', loss, on_epoch=True)

        figure = warp_figure(
            img=batch['img'][0].squeeze().numpy(), 
            uv=batch['uv'][0].squeeze().numpy(), 
            duv=batch['duv'][0].squeeze().cpu().numpy(), 
            duv_pred=duv_pred[0].squeeze().cpu().numpy(), 
            H=batch['H'][0].squeeze().numpy()
        )
        self.logger.experiment.add_figure('val_wrp', figure, self.global_step)
        return loss

    def test_step(self, batch, batch_idx):
        duv_pred = self.model(batch['img_seq'][0], batch['img_seq'][1])
        loss = self.loss(
            duv_pred.view(-1, 2), 
            batch['duv'].to(duv_pred.dtype).view(-1, 2)
        ).mean()
        self.log('test_loss', loss)

        figure = warp_figure(
            img=batch['img'][0].squeeze().numpy(), 
            uv=batch['uv'][0].squeeze().numpy(), 
            duv=batch['duv'][0].squeeze().cpu().numpy(), 
            duv_pred=duv_pred[0].squeeze().cpu().numpy(), 
            H=batch['H'][0].squeeze().numpy()
        )
        self.logger.experiment.add_figure('test_wrp', figure, self.global_step)
        return loss