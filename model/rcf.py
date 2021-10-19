"""
@File : rcf.py
@Author : CodeCat
@Time : 2021/7/28 下午7:37
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Down_Basic(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(Down_Basic, self).__init__()
        self.conv1_1 = ConvBNRelu(in_channels, out_channels, 3)
        self.conv1_2 = ConvBNRelu(out_channels, 21, 1)

        self.conv2_1 = ConvBNRelu(out_channels, out_channels, 3)
        self.conv2_2 = ConvBNRelu(out_channels, 21, 1)

        self.conv = nn.Conv2d(21, num_classes, 1)

    def forward(self, inputs):
        x = self.conv1_1(inputs)
        x_l1 = self.conv1_2(x)
        x = self.conv2_1(x)
        x_l2 = self.conv2_2(x)
        x_l = x_l1 + x_l2
        x_l = self.conv(x_l)
        return x, x_l


class Down_Bottle(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(Down_Bottle, self).__init__()

        self.conv1_1 = ConvBNRelu(in_channels, out_channels, 3)
        self.conv1_2 = ConvBNRelu(out_channels, 21, 1)

        self.conv2_1 = ConvBNRelu(out_channels, out_channels, 3)
        self.conv2_2 = ConvBNRelu(out_channels, 21, 1)

        self.conv3_1 = ConvBNRelu(out_channels, out_channels, 3)
        self.conv3_2 = ConvBNRelu(out_channels, 21, 1)

        self.conv = nn.Conv2d(21, num_classes, 1)

    def forward(self, inputs):
        x = self.conv1_1(inputs)
        x_l1 = self.conv1_2(x)
        x = self.conv2_1(x)
        x_l2 = self.conv2_2(x)
        x = self.conv3_1(x)
        x_l3 = self.conv3_2(x)
        x_l = x_l1 + x_l2 + x_l3
        x_l = self.conv(x_l)
        return x, x_l


class RCF(nn.Module):
    # https://arxiv.org/abs/1612.02103
    def __init__(self, num_classes):
        super(RCF, self).__init__()

        self.stage1 = Down_Basic(3, 64, num_classes)
        self.stage2 = Down_Basic(64, 128, num_classes)
        self.stage3 = Down_Bottle(128, 256, num_classes)
        self.stage4 = Down_Bottle(256, 512, num_classes)
        self.stage5 = Down_Bottle(512, 512, num_classes)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv = nn.Conv2d(5 * num_classes, num_classes, 1)

    def forward(self, inputs):
        x, x_l1 = self.stage1(inputs)
        x = self.pool(x)

        x, x_l2 = self.stage2(x)
        x_l2 = F.interpolate(x_l2, size=inputs.shape[2:], mode='bilinear', align_corners=True)
        x = self.pool(x)

        x, x_l3 = self.stage3(x)
        x_l3 = F.interpolate(x_l3, size=inputs.shape[2:], mode='bilinear', align_corners=True)
        x = self.pool(x)

        x, x_l4 = self.stage4(x)
        x_l4 = F.interpolate(x_l4, size=inputs.shape[2:], mode='bilinear', align_corners=True)
        x = self.pool(x)

        x, x_l5 = self.stage5(x)
        x_l5 = F.interpolate(x_l5, size=inputs.shape[2:], mode='bilinear', align_corners=True)

        out = [x_l1, x_l2, x_l3, x_l4, x_l5]
        out = torch.cat(out, dim=1)
        out = self.conv(out)
        return out, x_l5, x_l4, x_l3, x_l2, x_l1


if __name__ == '__main__':
    inp = torch.randn((2, 3, 512, 512))

    model = RCF(2)

    out = model(inp)

    for i in range(len(out)):
        print(out[i].shape)
