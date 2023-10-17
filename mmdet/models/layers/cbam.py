import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.inter_channels = in_planes // ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, self.inter_channels, 1, bias=False),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            nn.Conv2d(self.inter_channels, in_planes, 1, bias=False),
            nn.BatchNorm2d(in_planes)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_pool = self.mlp(self.avg_pool(x))
        attention = self.sigmoid(avg_pool + max_out)
        return attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size = 7) -> None:
        super().__init__()
        padding = 3 if kernel_size == 7 else 1

        ## The input is from ChannelAttention which is avg + max
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding),
            nn.BatchNorm2d(1)
        )

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.spatial_attention(out)
        attention = self.sigmoid(out)
        return attention

class CBAM(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    """
    ex:
    x = torch.randn(3, 64, 56, 56)
​
    channel = channel_attention(x)  # (3, 64, 1, 1)
    x = channel * x                 # (3, 64, 56, 56)
    ​
    spatial = spatial_attention(x)  # (3, 1, 56, 56)
    x = spatial * x                 # (3, 64, 56, 56)
    """

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out