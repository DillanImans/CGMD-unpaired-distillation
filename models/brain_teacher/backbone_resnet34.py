import torch.nn as nn


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = lambda c: nn.InstanceNorm3d(c, affine=True)
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.norm1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.norm2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet34Backbone(nn.Module):
    def __init__(self, in_ch: int = 1, feat_dim: int = 512, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = lambda c: nn.InstanceNorm3d(c, affine=True)
        self.inplanes = 64

        self.conv1 = nn.Conv3d(
            in_ch,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.norm1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock3D, 64, 3, norm_layer=norm_layer)
        self.layer2 = self._make_layer(BasicBlock3D, 128, 4, stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(BasicBlock3D, 256, 6, stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(BasicBlock3D, 512, 3, stride=2, norm_layer=norm_layer)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.feat_dim = feat_dim
        assert feat_dim == 512, "ResNet34Backbone output channels are fixed to 512."

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = lambda c: nn.InstanceNorm3d(c, affine=True)

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x.flatten(1)
