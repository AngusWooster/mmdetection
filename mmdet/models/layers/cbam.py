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

    channel = channel_attention(x)  # (3, 64, 1, 1)
    x = channel * x                 # (3, 64, 56, 56)

    spatial = spatial_attention(x)  # (3, 1, 56, 56)
    x = spatial * x                 # (3, 64, 56, 56)
    """

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class DepthWise_SeparableConv(nn.Module):
    def __init__(self, n_in, n_out) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(n_in, n_in, 3, padding=1, groups=n_in)
        self.pointwise = nn.Conv2d(n_in, n_out, 1)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class ConvAdapter(nn.Module):
    def __init__(self, in_planes, ratio=2) -> None:
        super().__init__()
        self.inter_channels = in_planes // ratio
        self.conv1 = DepthWise_SeparableConv(in_planes, self.inter_channels)
        self.bn1 = nn.BatchNorm2d(self.inter_channels)
        self.activate = nn.ReLU(inplace=True)
        self.conv2 = DepthWise_SeparableConv(self.inter_channels, in_planes)
        self.bn2 = nn.BatchNorm2d(in_planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activate(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out


class CABAM(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()
        self.conv_adapter = ConvAdapter(channels)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        xt = self.conv_adapter(x)

        return out + xt

if __name__ == '__main__':
    in_channel = 256
    dropout = 0

    cbam = CBAM(256)
    paremeters = sum(p.numel() for p in cbam.parameters() if p.requires_grad)
    print(f"paremeters: {paremeters}")

    ma_cbam = CABAM(256)
    paremeters = sum(p.numel() for p in ma_cbam.parameters() if p.requires_grad)
    print(f"paremeters: {paremeters}")


    x = torch.rand((2, in_channel, 56, 56))
    print(f"Input: {x.shape}")
    out = ma_cbam(x)
    print(f"out: {out.shape}")
