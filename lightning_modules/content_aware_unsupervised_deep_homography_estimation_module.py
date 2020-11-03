import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
from collections import OrderedDict
import pytorch_lightning as pl
from typing import List

from models import DeepHomographyRegression
from models import ConvBlock


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
            ('conv5', ConvBlock(32, 1, padding=1)), # TODO: logits>>
        ]))
        self.homography_estimator = resnet34(pretrained=False)
        self.homography_estimator.conv1.in_channels = shape[0]
        self.homography_estimator.fc.out_features = 8

        self.lam = lam
        self.mu = mu
        self.lr = lr
        self.betas = betas

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)
        return optimizer

    def forward(self, img_a, img_b):
        # features
        f_a = self.feature_extractor(img_a)
        f_b = self.feature_extractor(img_b)

        # masks
        m_a = self.mask_predictor(img_a)
        m_b = self.mask_predictor(img_b)

        # weighted feature maps
        g_a = m_a.mul(f_a)
        g_b = m_b.mul(f_b)

        h_ab = torch.cat((g_a, g_b), dim=1) # BxCxHxW
        h_ab = self.homography_estimator(h_ab)

        return {
            'h_ab': h_ab,
            'f_a': f_a,
            'f_b': f_b,
            'm_a': m_a,
            'm_b': m_b
        }

    def content_loss(self, f_a: torch.Tensor, f_b: torch.Tensor, m_a: torch.Tensor, m_b: torch.Tensor):
        eps = torch.finfo(f_a.dtype).eps
        loss = torch.sum(m_a.mul(m_b).mul(torch.abs(f_a.sub(f_b)))).div(torch.sum(m_a.mul(m_b) + eps))
        return loss

    def regularizer_loss(self, f_a: torch.Tensor, f_b: torch.Tensor):
        loss = F.l1_loss(f_a, f_b)
        return loss

    def consistency_loss(self, h_ab: torch.Tensor, h_ba: torch.Tensor):
        loss = F.mse_loss(h_ab.matmul(h_ba), torch.eye(3, dtype=h_ab.dtype, device=h_ab.device))
        return loss

    def training_step(self, batch, batch_idx):
        # forward ab and ba
        i_a = batch['img_seq'][0]
        i_b = batch['img_seq'][1]

        ab_dic = self(i_a, i_b)
        ba_dic = self(i_b, i_a)

        # warp images
        

        # regularizer
        # consistency

        # l_content_ab = self.content_loss()
        # l_content_ba = self.content_loss()
        # l_reg = self.regularizer_loss(f_a, f_b)
        # l_consistency = self.consistency_loss(h_ab, h_ba)
        # loss = l_content_ab + l_content_ba - self.lam*l_reg + self.mu*l_consistency

        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass
