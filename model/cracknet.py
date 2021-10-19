"""
@File : cracknet.py
@Author : CodeCat
@Time : 2021/8/30 下午7:17
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1)//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CrackNet(nn.Module):
    # https://ieeexplore.ieee.org/abstract/document/8961534
    def __init__(self, num_classes):
        super(CrackNet, self).__init__()

        self.conv1_1 = ConvReLU(3, 64, 3)
        self.conv1_2 = ConvReLU(64, 64, 3)
        self.conv1 = nn.Conv2d(64, num_classes, 1)

        self.conv2_1 = ConvReLU(64, 128, 3)
        self.conv2_2 = ConvReLU(128, 128, 3)
        self.conv2 = nn.Conv2d(128, num_classes, 1)

        self.conv3_1 = ConvReLU(128, 256, 3)
        self.conv3_2 = ConvReLU(256, 256, 3)
        self.conv3_3 = ConvReLU(256, 256, 3)
        self.conv3 = nn.Conv2d(256, num_classes, 1)

        self.conv4_1 = ConvReLU(256, 512, 3)
        self.conv4_2 = ConvReLU(512, 512, 3)
        self.conv4_3 = ConvReLU(512, 512, 3)
        self.conv4 = nn.Conv2d(512, num_classes, 1)

        self.conv5_1 = ConvReLU(512, 512, 3)
        self.conv5_2 = ConvReLU(512, 512, 3)
        self.conv5_3 = ConvReLU(512, 512, 3)
        self.conv5 = nn.Conv2d(512, num_classes, 1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifer = nn.Conv2d(5*num_classes, num_classes, 1)

    def forward(self, x):
        h, w = x.shape[2:]

        x = self.conv1_2(self.conv1_1(x))
        x1 = self.conv1(x)
        out1 = x1

        x = self.pool(x)
        x = self.conv2_2(self.conv2_1(x))
        x2 = self.conv2(x)
        out2 = F.interpolate(x2, size=(h, w), mode='bilinear', align_corners=True)

        x = self.pool(x)
        x = self.conv3_3(self.conv3_2(self.conv3_1(x)))
        x3 = self.conv3(x)
        out3 = F.interpolate(x3, size=(h, w), mode='bilinear', align_corners=True)

        x = self.pool(x)
        x = self.conv4_3(self.conv4_2(self.conv4_1(x)))
        x4 = self.conv4(x)
        out4 = F.interpolate(x4, size=(h, w), mode='bilinear', align_corners=True)

        x = self.pool(x)
        x = self.conv5_3(self.conv5_2(self.conv5_1(x)))
        x5 = self.conv5(x)
        out5 = F.interpolate(x5, size=(h, w), mode='bilinear', align_corners=True)

        out = torch.cat([out1, out2, out3, out4, out5], dim=1)
        out = self.classifer(out)
        return out, out5, out4, out3, out2, out1


if __name__ == '__main__':
    input = torch.rand(1, 3, 384, 384)
    model = CrackNet(2)
    out, out_1, out_2, out_3, out_4, out_5 = model(input)
    print(out.shape)
    print(out_1.shape)
    print(out_2.shape)
    print(out_3.shape)
    print(out_4.shape)
    print(out_5.shape)