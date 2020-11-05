import torch
import torch.nn as nn
from collections import OrderedDict

from models import ConvBlock


class DeepHomographyRegression(nn.Module):
    r"""Homography regression model from https://arxiv.org/pdf/1606.03798.pdf.

    Args:
        shape (tuple of int): Input shape of single image, CxHxW.
    """
    def __init__(self, shape):
        super(DeepHomographyRegression, self).__init__()
        self.features = nn.Sequential(OrderedDict([
                ('conv1', ConvBlock(2*shape[0], 64)),
                ('conv2', ConvBlock(64, 64)),
                ('mp1', nn.MaxPool2d(kernel_size=2, stride=2)),
                ('conv3', ConvBlock(64, 64)),
                ('conv4', ConvBlock(64, 64)),
                ('mp2', nn.MaxPool2d(kernel_size=2, stride=2)),
                ('conv5', ConvBlock(64, 128)),
                ('conv6', ConvBlock(128, 128)),
                ('mp3', nn.MaxPool2d(kernel_size=2, stride=2)),
                ('conv7', ConvBlock(128, 128)),
                ('conv8', ConvBlock(128, 128)),
                ('dropout', nn.Dropout2d(p=0.5)),
                ('faltten', nn.Flatten())
        ]))

        n = self._numel([1,2*shape[0],shape[1],shape[2]])

        self.regression = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(n, 1024)),
            ('fc2', nn.Linear(1024, 8))
        ]))

    def _numel(self, shape):
        x = torch.rand(shape)
        x = self.features(x)
        return x.numel()

    def forward(self, img, wrp):
        duv_pred = torch.cat([img, wrp], dim=1)
        duv_pred = self.features(duv_pred)
        duv_pred = self.regression(duv_pred)
        duv_pred = duv_pred.view(-1,4,2)
        return duv_pred
