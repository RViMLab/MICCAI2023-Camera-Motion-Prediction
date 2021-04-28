import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from kornia import warp_perspective, get_perspective_transform, crop_and_resize, tensor_to_image
from pycls.models import model_zoo
from typing import List

from utils.viz import warp_figure

class UnsupervisedDeepHomographyEstimationModuleBackbone(pl.LightningModule):
    
    def __init__(
        self, 
        shape: List[int], 
        pretrained: bool=False,
        lr: float=1e-4, 
        betas: List[float]=[0.9, 0.999], 
        milestones: List[int]=[0], 
        gamma: float=1.0, 
        log_n_steps: int=1000, 
        backbone: str='ResNet-34'
    ):
        super().__init__()
        self.save_hyperparameters('lr', 'betas', 'backbone')
        if backbone == 'ResNet-34':
            self.model = getattr(models, 'resnet34')(**{'pretrained': pretrained})

            # modify in and out layers
            self.model.conv1 = nn.Conv2d(
                in_channels=6,
                out_channels=self.model.conv1.out_channels,
                kernel_size=self.model.conv1.kernel_size,
                stride=self.model.conv1.stride,
                padding=self.model.conv1.padding
            )
            self.model.fc = nn.Linear(
                in_features=self.model.fc.in_features,
                out_features=8
            )
        else:
            if backbone not in model_zoo.get_model_list():
                raise ValueError('Model {} not available.'.format(backbone))
            self.model = model_zoo.build_model(backbone, pretrained)

            self.model.stem.conv = nn.Conv2d(
                in_channels=6,
                out_channels=self.model.stem.conv.out_channels,
                kernel_size=self.model.stem.conv.kernel_size,
                stride=self.model.stem.conv.stride,
                padding=self.model.stem.conv.padding
            )

            self.model.head.fc = nn.Linear(
                in_features=self.model.head.fc.in_features,
                out_features=8
            )

        self._mse_loss = nn.MSELoss()
        self._distance_loss = nn.PairwiseDistance()

        self._lr = lr
        self._betas = betas
        self._milestones = milestones
        self._gamma = gamma
        self._validation_step_ct = 0
        self._log_n_steps = log_n_steps

    def forward(self, img, wrp):
        cat = torch.cat((img, wrp), dim=1)
        return self._model(cat).view(-1,4,2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr, betas=self._betas)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self._milestones, gamma=self._gamma)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        duv_pred = self(batch['img_crp'], batch['wrp_crp'])

        uv_wrp = batch['uv'] + duv_pred
        H_pred = get_perspective_transform(batch['uv'].to(duv_pred.dtype).flip(-1), uv_wrp.flip(-1))
        wrp_pred = warp_perspective(batch['img_pair'][0], torch.inverse(H_pred), batch['img_pair'][0].shape[-2:])
        wrp_crp_pred = crop_and_resize(wrp_pred, batch['uv'].flip(-1), batch['wrp_crp'].shape[-2:])

        mse_loss = self._mse_loss(wrp_crp_pred, batch['wrp_crp'])
        distance_loss = self._distance_loss(
            duv_pred.view(-1, 2), 
            batch['duv'].to(duv_pred.dtype).view(-1, 2)
        ).mean()

        self.log('train/mse_loss', mse_loss)
        self.log('train/distance', distance_loss)

        return mse_loss

    def validation_step(self, batch, batch_idx):
        duv_pred = self(batch['img_crp'], batch['wrp_crp'])
        distance_loss = self._distance_loss(
            duv_pred.view(-1, 2), 
            batch['duv'].to(duv_pred.dtype).view(-1, 2)
        ).mean()
        self.log('val/distance', distance_loss, on_epoch=True)

        if self._validation_step_ct % self._log_n_steps == 0:
            figure = warp_figure(
                img=tensor_to_image(batch['img_pair'][0][0]), 
                uv=batch['uv'][0].squeeze().cpu().numpy(), 
                duv=batch['duv'][0].squeeze().cpu().numpy(), 
                duv_pred=duv_pred[0].squeeze().cpu().numpy(), 
                H=batch['H'][0].squeeze().numpy()
            )
            self.logger.experiment.add_figure('val/wrp', figure, self._validation_step_ct)
        self._validation_step_ct += 1
        return distance_loss

    def test_step(self, batch, batch_idx):
        duv_pred = self(batch['img_crp'], batch['wrp_crp'])
        distance_loss = self._distance_loss(
            duv_pred.view(-1, 2), 
            batch['duv'].to(duv_pred.dtype).view(-1, 2)
        ).mean()
        self.log('test/distance', distance_loss, on_epoch=True)
        return distance_loss
