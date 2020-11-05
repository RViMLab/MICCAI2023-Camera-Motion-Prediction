import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
from collections import OrderedDict
import pytorch_lightning as pl
from typing import List
from kornia.geometry.transform import get_perspective_transform, warp_perspective, crop_and_resize

from models import DeepHomographyRegression
from models import ConvBlock
from utils.viz import warp_figure


class ContentAwareUnsupervisedDeepHomographyEstimationModule(pl.LightningModule):
    def __init__(self, shape: List[int], lam: float=2.0, mu: float=0.01, lr: float=1e-4, betas: List[float]=[0.9, 0.999]):
        r"""Content-aware unsupervised deep homography estimation model from https://arxiv.org/abs/1909.05983.

        Args:
            shape (tuple of int): Input shape CxHxW.
        """
        super().__init__()

        self.save_hyperparameters('lam', 'mu', 'lr', 'betas')

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', ConvBlock(shape[0], 4, padding=1)),  # preserve dimensions
            ('conv2', ConvBlock(4, 8, padding=1)),
            ('conv3', ConvBlock(8, 1, padding=1))
        ]))
        self.mask_predictor = nn.Sequential(OrderedDict([
            ('conv1', ConvBlock(shape[0], 4, padding=1)),
            ('conv2', ConvBlock(4, 8, padding=1)),
            ('conv3', ConvBlock(8, 16, padding=1)),
            ('conv4', ConvBlock(16, 32, padding=1)),
            ('conv5', ConvBlock(32, 1, padding=1, activation=torch.sigmoid)),
        ]))
        self.homography_estimator = resnet34(pretrained=False)
        # modify in and out layers
        self.homography_estimator.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=self.homography_estimator.conv1.out_channels,
            kernel_size=self.homography_estimator.conv1.kernel_size,
            stride=self.homography_estimator.conv1.stride,
            padding=self.homography_estimator.conv1.padding
        )
        self.homography_estimator.fc = nn.Linear(
            in_features=self.homography_estimator.fc.in_features,
            out_features=8
        )

        self.lam = lam
        self.mu = mu
        self.lr = lr
        self.betas = betas

        self.val_loss = nn.PairwiseDistance()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)
        return optimizer

    def forward(self, img_0, img_1):
        # features
        f_0 = self.feature_extractor(img_0)
        f_1 = self.feature_extractor(img_1)

        # masks
        m_0 = self.mask_predictor(img_0)
        m_1 = self.mask_predictor(img_1)

        # weighted feature maps
        g_0 = m_0.mul(f_0)
        g_1 = m_1.mul(f_1)

        duv_01 = torch.cat((g_0, g_1), dim=1) # BxCxHxW
        duv_01 = self.homography_estimator(duv_01)
        duv_01 = duv_01.view(-1,4,2)

        return {
            'duv_01': duv_01,
            'f_0': f_0,
            'f_1': f_1,
            'm_0': m_0,
            'm_1': m_1
        }

    def content_loss(self, f_0: torch.Tensor, f_1: torch.Tensor, m_0: torch.Tensor, m_1: torch.Tensor):
        r"""Computes content loss, see paper eq. 4.

        Args:
            f_0 (torch.Tensor): Feature map for image 0
            f_1 (torch.Tensor): Feature map for image 1
            m_0 (torch.Tensor): Mask for image 0
            m_1 (torch.Tensor): Mask for image 1

        Return:
            loss (torch.Tensor): Loss
        """
        eps = torch.finfo(f_0.dtype).eps
        loss = torch.sum(m_0.mul(m_1).mul(torch.abs(f_0.sub(f_1)))).div(torch.sum(m_0.mul(m_1) + eps))
        return loss

    def regularizer_loss(self, f_0: torch.Tensor, f_1: torch.Tensor):
        r"""Computes regularizer, see paper eq. 5.

        Args:
            f_0 (torch.Tensor): Feature map for image 0
            f_1 (torch.Tensor): Feature map for image 1

        Return:
            loss (torch.Tensor): L1 loss
        """
        loss = F.l1_loss(f_0, f_1)
        return loss

    def consistency_loss(self, h_01: torch.Tensor, h_21: torch.Tensor):
        r"""Computes forward backward consistency loss, see paper eq. 6.

        Args:
            h_01 (torch.Tensor): Homography matrix of shape 3x3 from 0 to 1
            h_21 (torch.Tensor): Homography matrix of shape 3x3 from 1 to 0

        Return:
            loss (torch.Tensor): MSE loss
        """
        identity = torch.eye(3, dtype=h_01.dtype, device=h_01.device).reshape((1,3,3)).repeat(h_01.shape[0],1,1)
        loss = F.mse_loss(h_01.matmul(h_21), identity)
        return loss

    def four_pt_to_matrix_homography_representation(self, uv_0: torch.Tensor, duv_01: torch.Tensor):
        r"""Transforms homography from four point representation of shape 4x2 to matrix representation of shape 3x3.

        Args:
            uv_0 (torch.Tensor): Image edges in image coordinates
            duv_01 (torch.Tensor): Deviation from edges in image coordinates
        """
        uv_b = uv_0 + uv_0
        h_01 = get_perspective_transform(uv_0.flip(-1), uv_b.flip(-1))
        return h_01

    def training_step(self, batch, batch_idx):
        # forward ab and ba
        i_a = batch['img_seq_crp'][0]
        i_b = batch['img_seq_crp'][1] # warped and cropped

        ab_dic = self(i_a, i_b)
        ba_dic = self(i_b, i_a)

        # warp images and masks
        h_ab = self.four_pt_to_matrix_homography_representation(batch['uv'].to(ab_dic['duv_01'].dtype), ab_dic['duv_01'])
        h_ba = self.four_pt_to_matrix_homography_representation(batch['uv'].to(ba_dic['duv_01'].dtype), ba_dic['duv_01'])

        i_a_prime = warp_perspective(i_a, torch.inverse(h_ab), i_a.shape[-2:])
        i_b_prime = warp_perspective(i_b, torch.inverse(h_ba), i_a.shape[-2:])

        m_a_prime = warp_perspective(ab_dic['m_0'], torch.inverse(h_ab), ab_dic['m_0'].shape[-2:])
        m_b_prime = warp_perspective(ba_dic['m_0'], torch.inverse(h_ba), ba_dic['m_0'].shape[-2:])

        # compute losses
        l_content_ab = self.content_loss(self.feature_extractor(i_a_prime), ab_dic['f_1'], m_a_prime, ab_dic['m_1'])
        l_content_ba = self.content_loss(self.feature_extractor(i_b_prime), ba_dic['f_1'], m_b_prime, ba_dic['m_1'])
        l_reg = self.regularizer_loss(ab_dic['f_0'], ab_dic['f_1'])
        l_consistency = self.consistency_loss(h_ab, h_ba)
        loss = l_content_ab + l_content_ba - self.lam*l_reg + self.mu*l_consistency

        return loss

    def validation_step(self, batch, batch_idx):
        dic = self(batch['img_seq_crp'][0], batch['img_seq_crp'][1])
        loss = self.val_loss(
            dic['duv_01'].view(-1, 2), 
            batch['duv'].to(dic['duv_01'].dtype).view(-1, 2)
        ).mean()
        self.log('val_loss', loss, on_epoch=True)

        wrp_figure = warp_figure(
            img=batch['img_seq'][0].squeeze().cpu().numpy(), 
            uv=batch['uv'][0].squeeze().cpu().numpy(), 
            duv=batch['duv'][0].squeeze().cpu().numpy(), 
            duv_pred=dic['duv_01'][0].squeeze().cpu().numpy(), 
            H=batch['H'][0].squeeze().numpy()
        )

        self.logger.experiment.add_figure('val_wrp', wrp_figure, self.current_epoch)
        self.logger.experiment.add_images('val/img_seq_crp_0', batch['img_seq_crp'][0], self.current_epoch)
        self.logger.experiment.add_images('val/img_seq_crp_1', batch['img_seq_crp'][1], self.current_epoch)
        self.logger.experiment.add_images('val/mask_0', dic['m_0'], self.current_epoch)
        self.logger.experiment.add_images('val/mask_1', dic['m_1'], self.current_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        duv_pred = self.model(batch['img_seq_crp'][0], batch['img_seq_crp'][1])['duv_01']
        loss = self.loss(
            duv_pred.view(-1, 2), 
            batch['duv'].to(duv_pred.dtype).view(-1, 2)
        ).mean()
        self.log('test_loss', loss)
        return loss
