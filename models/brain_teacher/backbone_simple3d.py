import torch.nn as nn


class Simple3DBackbone(nn.Module):
    def __init__(self, in_ch: int = 1, feat_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 16, 3, stride = 2, padding = 1),
            nn.InstanceNorm3d(16),
            nn.ReLU(inplace = True),

            nn.Conv3d(16, 32, 3, stride = 2, padding = 1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace = True),

            nn.Conv3d(32, 64, 3, stride = 2, padding = 1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace = True),

            nn.Conv3d(64, 128, 3, stride = 2, padding = 1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace = True),

            nn.AdaptiveAvgPool3d(1),
        )
        self.feat_dim = feat_dim
        assert feat_dim == 128, "If we change channels, we need to update feat_dim"

    def forward(self, x):
        x = self.net(x)
        return x.flatten(1)