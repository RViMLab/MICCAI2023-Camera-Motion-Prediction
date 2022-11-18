import torch
import torchvision


class VarResNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_features: int,
        resnet: str="resnet18",
        pretrained: bool=False
    ) -> None:
        r"""Creates a ResNet from torchvision.models with
        variable input and output features.

        Args:
            in_channels (int): Number of input channels
            out_features (int): Number of output classes
            resnet (str): Name of ResNet, e.g. [resnet18/34/50/101/152]
            pretrained (bool): Whether to load pre-trained
        """
        super().__init__()
        self._model = getattr(torchvision.models, resnet)(
            pretrained=pretrained
        )

        self._model.conv1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=self._model.conv1.out_channels,
            kernel_size=self._model.conv1.kernel_size,
            stride=self._model.conv1.stride,
            padding=self._model.conv1.padding
        )

        self._model.fc = torch.nn.Linear(
            in_features=self._model.fc.in_features,
            out_features=out_features
        )

    def forward(self, x):
        return self._model(x)


if __name__ == "__main__":
    B = 2
    in_channels = 4
    out_channels = 128
    x = torch.ones([B, 4, 256, 256])
    model = VarResNet(4, 128, "resnet18")
    y = model(x)

    if y.shape[1] != out_channels:
        raise ValueError("Unexpected output shape.")
