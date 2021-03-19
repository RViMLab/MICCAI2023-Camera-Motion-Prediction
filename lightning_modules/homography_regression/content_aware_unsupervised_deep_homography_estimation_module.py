import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
from collections import OrderedDict
import pytorch_lightning as pl
from typing import List
from kornia import get_perspective_transform, warp_perspective, crop_and_resize, tensor_to_image

from models import DeepHomographyRegression
from models import ConvBlock
from utils.viz import warp_figure, yt_alpha_blend
from utils.processing import image_edges, four_point_homography_to_matrix


class ContentAwareUnsupervisedDeepHomographyEstimationModule(pl.LightningModule):
    def __init__(self, shape: List[int], lam: float=2.0, mu: float=0.01, pre_train_epochs: int=4, lr: float=1e-4, betas: List[float]=[0.9, 0.999], milestones: List[int]=[0], gamma: float=1.0, log_n_steps: int=1000):
        r"""Content-aware unsupervised deep homography estimation model from https://arxiv.org/abs/1909.05983.

        Args:
            shape (tuple of int): Input shape CxHxW.
        """
        super().__init__()

        self.save_hyperparameters('pre_train_epochs', 'lam', 'mu', 'lr', 'betas')

        self._feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', ConvBlock(shape[0], 4, padding=1)),  # preserve dimensions
            ('conv2', ConvBlock(4, 8, padding=1)),
            ('conv3', ConvBlock(8, 1, padding=1))
        ]))
        self._mask_predictor = nn.Sequential(OrderedDict([
            ('conv1', ConvBlock(shape[0], 4, padding=1)),
            ('conv2', ConvBlock(4, 8, padding=1)),
            ('conv3', ConvBlock(8, 16, padding=1)),
            ('conv4', ConvBlock(16, 32, padding=1)),
            ('conv5', ConvBlock(32, 1, padding=1, activation=torch.sigmoid)),
        ]))
        self._homography_estimator = resnet34(pretrained=False)

        # modify in and out layers
        self._homography_estimator.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=self._homography_estimator.conv1.out_channels,
            kernel_size=self._homography_estimator.conv1.kernel_size,
            stride=self._homography_estimator.conv1.stride,
            padding=self._homography_estimator.conv1.padding
        )
        self._homography_estimator.fc = nn.Linear(
            in_features=self._homography_estimator.fc.in_features,
            out_features=8
        )

        self._lam = lam
        self._mu = mu
        self._pre_train_epochs = pre_train_epochs
        self._lr = lr
        self._betas = betas
        self._milestones = milestones
        self._gamma = gamma

        self._distance_loss = nn.PairwiseDistance()
        self._validation_step_ct = 0
        self._log_n_steps = log_n_steps

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr, betas=self._betas)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self._milestones, gamma=self._gamma)
        return [optimizer], [scheduler]

    def forward(self, img_0, img_1, masks=True):
        # features
        f_0 = self._feature_extractor(img_0)
        f_1 = self._feature_extractor(img_1)

        # masks
        m_0 = self._mask_predictor(img_0)
        m_1 = self._mask_predictor(img_1)

        if masks:
            # weighted feature maps
            g_0 = m_0.mul(f_0)
            g_1 = m_1.mul(f_1)

            duv_01 = torch.cat((g_0, g_1), dim=1) # BxCxHxW

        else:
            duv_01 = torch.cat((f_0, f_1), dim=1) # BxCxHxW

        duv_01 = self._homography_estimator(duv_01)
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

    def four_point_homography_to_matrix(self, uv_0: torch.Tensor, duv_01: torch.Tensor):
        r"""Transforms homography from four point representation of shape 4x2 to matrix representation of shape 3x3.

        Args:
            uv_0 (torch.Tensor): Image edges in image coordinates
            duv_01 (torch.Tensor): Deviation from edges in image coordinates
        """
        uv_1 = uv_0 + duv_01
        h_01 = get_perspective_transform(uv_0.flip(-1), uv_1.flip(-1))
        return h_01

    def training_step(self, batch, batch_idx):
        # forward ab and ba
        i_a = batch['img_crp']
        i_b = batch['img_wrp'] # warped and cropped

        masks = True
        if self.current_epoch < self._pre_train_epochs:
            masks = False

        ab_dic = self(i_a, i_b, masks)
        ba_dic = self(i_b, i_a, masks)

        # warp images and masks
        h_ab = self.four_point_homography_to_matrix(batch['uv'].to(ab_dic['duv_01'].dtype), ab_dic['duv_01'])
        h_ba = self.four_point_homography_to_matrix(batch['uv'].to(ba_dic['duv_01'].dtype), ba_dic['duv_01'])

        i_a_prime = warp_perspective(i_a, torch.inverse(h_ab), i_a.shape[-2:])
        i_b_prime = warp_perspective(i_b, torch.inverse(h_ba), i_b.shape[-2:])

        m_a_prime = warp_perspective(ab_dic['m_0'], torch.inverse(h_ab), ab_dic['m_0'].shape[-2:])
        m_b_prime = warp_perspective(ba_dic['m_0'], torch.inverse(h_ba), ba_dic['m_0'].shape[-2:])

        # compute losses
        l_content_ab = self.content_loss(self._feature_extractor(i_a_prime), ab_dic['f_1'], m_a_prime, ab_dic['m_1'])
        l_content_ba = self.content_loss(self._feature_extractor(i_b_prime), ba_dic['f_1'], m_b_prime, ba_dic['m_1'])
        l_reg = self.regularizer_loss(ab_dic['f_0'], ab_dic['f_1'])
        l_consistency = self.consistency_loss(h_ab, h_ba)
        loss = l_content_ab + l_content_ba - self._lam*l_reg + self._mu*l_consistency

        # track losses
        distance_loss = self._distance_loss(
            ab_dic['duv_01'].view(-1, 2), 
            batch['duv'].to(ab_dic['duv_01'].dtype).view(-1, 2)
        ).mean()
        self.log('train/distance', distance_loss)

        self.log('train/content_loss_ab', l_content_ab)
        self.log('train/content_loss_ba', l_content_ba)
        self.log('train/regularizer_loss', l_reg)
        self.log('train/consistency_loss', l_consistency)
        self.log('train/composite_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        masks = True
        if self.current_epoch < self._pre_train_epochs:
            masks = False

        dic = self(batch['img_crp'], batch['wrp_crp'], masks)
        distance_loss = self._distance_loss(
            dic['duv_01'].view(-1, 2), 
            batch['duv'].to(dic['duv_01'].dtype).view(-1, 2)
        ).mean()
        self.log('val/distance', distance_loss, on_epoch=True)

        if self._validation_step_ct % self._log_n_steps == 0:
            # # log figures
            # uv = image_edges(batch['img_crp'])
            # H = four_point_homography_to_matrix(uv, duv_pred)
            # wrp_pred = warp_perspective(batch['img_crp'], torch.inverse(H), batch['wrp_crp'].shape[-2:])

            # blend = yt_alpha_blend(
            #     batch['wrp_crp'][0],
            #     wrp_pred[0]     
            # )

            wrp_figure = warp_figure(
                img=tensor_to_image(batch['img_pair'][0][0]), 
                uv=batch['uv'][0].squeeze().cpu().numpy(), 
                duv=batch['duv'][0].squeeze().cpu().numpy(), 
                duv_pred=dic['duv_01'][0].squeeze().cpu().numpy(), 
                H=batch['H'][0].squeeze().numpy()
            )

            self.logger.experiment.add_figure('val/wrp', wrp_figure, self._validation_step_ct)
            # self.logger.experiment.add_image('val/blend', blend, self._validation_step_ct)
            self.logger.experiment.add_images('val/img_crp', batch['img_crp'], self._validation_step_ct)
            self.logger.experiment.add_images('val/wrp_crp', batch['wrp_crp'], self._validation_step_ct)
            self.logger.experiment.add_images('val/mask_0', dic['m_0'], self._validation_step_ct)
            self.logger.experiment.add_images('val/mask_1', dic['m_1'], self._validation_step_ct)
        self._validation_step_ct += 1
        return distance_loss

    def test_step(self, batch, batch_idx):
        duv_pred = self(batch['img_crp'], batch['wrp_crp'], masks=True)['duv_01']
        distance_loss = self._distance_loss(
            duv_pred.view(-1, 2), 
            batch['duv'].to(duv_pred.dtype).view(-1, 2)
        ).mean()
        self.log('test/distance', distance_loss, on_epoch=True)

        return distance_loss
