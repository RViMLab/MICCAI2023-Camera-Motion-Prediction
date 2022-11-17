from collections import OrderedDict

import torch

from models import ConvBlock, DeConvBlock


class CompletionNetwork(torch.nn.Module):
    def __init__(self, in_channels: int, features: int=64) -> None:
        r"""Completion network as implemented in Globally and Locally Consistent Image Completion.
            http://iizuka.cs.tsukuba.ac.jp/projects/completion/data/completion_sig2017.pdf
        """
        if features % 2 != 0:
            raise ValueError("Hidden features must be divisible by 2.")
        super().__init__()
        self._downscale = torch.nn.Sequential(OrderedDict([
            ("conv_0", ConvBlock(in_channels, features  , 5, 1, 2, activation=torch.relu)),
            ("conv_1", ConvBlock(features   , features*2, 3, 2, 1, activation=torch.relu)),
            ("conv_2", ConvBlock(features*2 , features*2, 3, 1, 1, activation=torch.relu)),
        ]))

        self._bottleneck = torch.nn.Sequential(OrderedDict([
            ("conv_0", ConvBlock(features*2, features*4, 3, 2,  1, activation=torch.relu)),
            ("conv_1", ConvBlock(features*4, features*4, 3, 1,  1, activation=torch.relu)),
            ("conv_2", ConvBlock(features*4, features*4, 3, 1,  1, activation=torch.relu)),
            ("dil_0",  ConvBlock(features*4, features*4, 3, 1,  2, dilation=2,  activation=torch.relu)),
            ("dil_1",  ConvBlock(features*4, features*4, 3, 1,  4, dilation=4,  activation=torch.relu)),
            ("dil_2",  ConvBlock(features*4, features*4, 3, 1,  8, dilation=8,  activation=torch.relu)),
            ("dil_3",  ConvBlock(features*4, features*4, 3, 1, 16, dilation=16, activation=torch.relu)),
            ("conv_3", ConvBlock(features*4, features*4, 3, 1,  1, activation=torch.relu)),
            ("conv_4", ConvBlock(features*4, features*4, 3, 1,  1, activation=torch.relu)),
        ]))

        self._upscale = torch.nn.Sequential(OrderedDict([
            ("deconv_0", DeConvBlock(features*4, features*2      , 4, 2, 1, activation=torch.relu)),
            ("conv_0",     ConvBlock(features*2, features*2      , 3, 1, 1, activation=torch.relu)),
            ("deconv_1", DeConvBlock(features*2, features        , 4, 2, 1, activation=torch.relu)),
            ("conv_1",     ConvBlock(features  , int(features/2) , 3, 1, 1, activation=torch.relu)),
        ]))

        self._head = ConvBlock(int(features/2), in_channels, 3, 1, 1, activation=torch.sigmoid)

    def forward(self, x):
        x = self._downscale(x)
        x = self._bottleneck(x)
        x = self._upscale(x)
        return self._head(x)


if __name__ == "__main__":
    in_channels = 3
    input = torch.ones([1, in_channels, 120, 160])
    model = CompletionNetwork(in_channels, features=16)
    output = model(input)
    print(output.shape)
