"""
@File : my_att.py
@Author : CodeCat
@Time : 2021/8/6 上午11:08
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, in_planes):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, 1, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.sigmoid(x)


class SACAttention(nn.Module):
    def __init__(self, in_channels):
        super(SACAttention, self).__init__()
        self.ca = ChannelAttention(in_planes=in_channels)
        self.sa = SpatialAttention(in_planes=in_channels)

    def forward(self, x):
        x1 = self.ca(x) * x
        x2 = self.sa(x) * x
        return (x1 + x2) / 2


if __name__ == '__main__':
    input = torch.randn(1, 64, 484, 484)
    model = SACAttention(in_channels=64)
    out = model(input)
    print(out.shape)