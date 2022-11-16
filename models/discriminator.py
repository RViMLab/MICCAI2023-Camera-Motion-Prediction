from collections import OrderedDict
from typing import List

import torch

from models import ConvBlock


class Discriminator(torch.nn.Module):      
    def __init__(
        self, 
        in_channels: int,
        out_channels: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        negative_slope: float=0.2
    ) -> None:
        super().__init__()

        self._model = torch.nn.Sequential(OrderedDict([
            (
                "layer_0",
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels[0],
                    kernel_size=kernel_sizes[0],
                    stride=strides[0],
                    activation=torch.nn.LeakyReLU(negative_slope)
                )
            )
        ]))

        for idx, (out_channel, kernel_size, stride) in enumerate(zip(out_channels[1:], kernel_sizes[1:], strides[1:])):
            self._model.add_module(
                f"layer_{idx+1}",
                ConvBlock(
                    in_channels=out_channels[idx],
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    activation=torch.nn.LeakyReLU(negative_slope)
                )
            )

        self._model.add_module(
            f"adaptive_avg",
            torch.nn.AdaptiveAvgPool3d(
                1
            )
        )

    def forward(self, x):
        return torch.sigmoid(self._model(x))


if __name__ == "__main__":
    in_channels = 3
    d = Discriminator(
        in_channels,
        [32, 64, 128, 256, 1],
        [5, 5, 5, 5, 5],
        [2, 2, 2, 2, 1]
    )

    probe = torch.randn([1, in_channels, 128, 160])
    out = d(probe)
    print(out.shape)
