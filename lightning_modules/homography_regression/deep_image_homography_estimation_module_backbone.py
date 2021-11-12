import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from typing import List
from kornia.geometry import warp_perspective
from kornia import tensor_to_image
from pycls.models import model_zoo

from utils.viz import warp_figure, yt_alpha_blend
from utils.processing import image_edges, four_point_homography_to_matrix


class DeepImageHomographyEstimationModuleBackbone(pl.LightningModule):
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
        backbone_dict = {
            'ResNet-18': 'resnet18',
            'ResNet-34': 'resnet34'
        }
        if backbone == 'ResNet-18' or backbone == 'ResNet-34':
            backbone = backbone_dict[backbone]

        if backbone == 'resnet18' or backbone == 'resnet34':
            self._model = getattr(models, backbone)(**{'pretrained': pretrained})

            # modify in and out layers
            self._model.conv1 = nn.Conv2d(
                in_channels=6,
                out_channels=self._model.conv1.out_channels,
                kernel_size=self._model.conv1.kernel_size,
                stride=self._model.conv1.stride,
                padding=self._model.conv1.padding
            )
            self._model.fc = nn.Linear(
                in_features=self._model.fc.in_features,
                out_features=8
            )
        elif backbone == 'VGG':
            from models import DeepHomographyRegression

            self._model = DeepHomographyRegression(shape)
        else:
            if backbone not in model_zoo.get_model_list():
                raise ValueError('Model {} not available.'.format(backbone))
            self._model = model_zoo.build_model(backbone, pretrained)

            self._model.stem.conv = nn.Conv2d(
                in_channels=6,
                out_channels=self._model.stem.conv.out_channels,
                kernel_size=self._model.stem.conv.kernel_size,
                stride=self._model.stem.conv.stride,
                padding=self._model.stem.conv.padding
            )

            self._model.head.fc = nn.Linear(
                in_features=self._model.head.fc.in_features,
                out_features=8
            )

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
        duv_fw_pred = self(batch['img_crp'], batch['wrp_crp'])
        duv_bw_pred = self(batch['wrp_crp'], batch['img_crp'])
        distance_loss = self._distance_loss(
            duv_fw_pred.view(-1, 2), 
            batch['duv'].to(duv_fw_pred.dtype).view(-1, 2)
        ).mean()
        distance_loss += self._distance_loss(
            duv_bw_pred.view(-1, 2), 
            -batch['duv'].to(duv_bw_pred.dtype).view(-1, 2)
        ).mean()
        consistency_loss = self._distance_loss(
            duv_fw_pred.view(-1, 2),
            -duv_bw_pred.view(-1,2)
        ).mean()

        accumulated_loss = distance_loss + consistency_loss

        self.log('train/distance_loss', distance_loss) # logs all log_every_n_steps https://pytorch-lightning.readthedocs.io/en/latest/logging.html#control-logging-frequency
        self.log('train/consistency_loss', consistency_loss)
        self.log('train/accumulated_loss', accumulated_loss)
        return accumulated_loss

    def validation_step(self, batch, batch_idx):
        duv_fw_pred = self(batch['img_crp'], batch['wrp_crp'])
        duv_bw_pred = self(batch['wrp_crp'], batch['img_crp'])
        distance_loss = self._distance_loss(
            duv_fw_pred.view(-1, 2), 
            batch['duv'].to(duv_fw_pred.dtype).view(-1, 2)
        ).mean()
        distance_loss += self._distance_loss(
            duv_bw_pred.view(-1, 2), 
            -batch['duv'].to(duv_bw_pred.dtype).view(-1, 2)
        ).mean()
        consistency_loss = self._distance_loss(
            duv_fw_pred.view(-1, 2),
            -duv_bw_pred.view(-1,2)
        ).mean()

        accumulated_loss = distance_loss + consistency_loss

        self.log('val/distance_loss', distance_loss) # logs all log_every_n_steps https://pytorch-lightning.readthedocs.io/en/latest/logging.html#control-logging-frequency
        self.log('val/consistency_loss', consistency_loss)
        self.log('val/accumulated_loss', accumulated_loss)

        if self._validation_step_ct % self._log_n_steps == 0:
            # uv = image_edges(batch['img_crp'])
            # H = four_point_homography_to_matrix(uv, duv_pred)
            # wrp_pred = warp_perspective(batch['img_crp'], torch.inverse(H), batch['wrp_crp'].shape[-2:])

            # blend = yt_alpha_blend(
            #     batch['wrp_crp'][0],
            #     wrp_pred[0]     
            # )

            # self.logger.experiment.add_image('val/blend', blend, self._validation_step_ct)

            figure = warp_figure(
                img=tensor_to_image(batch['img_pair'][0][0]), 
                uv=batch['uv'][0].squeeze().cpu().numpy(), 
                duv=batch['duv'][0].squeeze().cpu().numpy(), 
                duv_pred=duv_fw_pred[0].squeeze().cpu().numpy(), 
                H=batch['H'][0].squeeze().cpu().numpy()
            )
            self.logger.experiment.add_figure('val/wrp', figure, self._validation_step_ct)
        self._validation_step_ct += 1

    def test_step(self, batch, batch_idx):
        duv_fw_pred = self(batch['img_crp'], batch['wrp_crp'])
        distance_loss = self._distance_loss(
            duv_fw_pred.view(-1, 2), 
            batch['duv'].to(duv_fw_pred.dtype).view(-1, 2)
        ).mean()
        self.log('test/distance_loss', distance_loss, on_epoch=True)
