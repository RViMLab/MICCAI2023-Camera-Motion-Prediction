from typing import Callable

import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t = 3,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        activation: Callable = torch.relu,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = activation

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class DeConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t = 3,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        activation: Callable = torch.relu,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = activation

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
