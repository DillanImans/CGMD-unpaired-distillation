import torch
import torch.nn as nn
from torchvision import models


class ResNet18Backbone2D(nn.Module):
    def __init__(self, in_ch: int = 1, feat_dim: int = 512, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)

        if in_ch != 3:
            conv1 = nn.Conv2d(
                in_ch,
                net.conv1.out_channels,
                kernel_size=net.conv1.kernel_size,
                stride=net.conv1.stride,
                padding=net.conv1.padding,
                bias=False,
            )
            if pretrained and in_ch == 1:
                with torch.no_grad():
                    conv1.weight.copy_(net.conv1.weight.mean(dim=1, keepdim=True))
            net.conv1 = conv1

        net.fc = nn.Identity()
        self.net = net
        self.feat_dim = feat_dim
        assert feat_dim == 512, "ResNet18Backbone2D output channels are fixed to 512."

    def forward(self, x):
        return self.net(x)


class ResNet34Backbone2D(nn.Module):
    def __init__(self, in_ch: int = 1, feat_dim: int = 512, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        net = models.resnet34(weights=weights)

        if in_ch != 3:
            conv1 = nn.Conv2d(
                in_ch,
                net.conv1.out_channels,
                kernel_size=net.conv1.kernel_size,
                stride=net.conv1.stride,
                padding=net.conv1.padding,
                bias=False,
            )
            if pretrained and in_ch == 1:
                with torch.no_grad():
                    conv1.weight.copy_(net.conv1.weight.mean(dim=1, keepdim=True))
            net.conv1 = conv1

        net.fc = nn.Identity()
        self.net = net
        self.feat_dim = feat_dim
        assert feat_dim == 512, "ResNet34Backbone2D output channels are fixed to 512."

    def forward(self, x):
        return self.net(x)
