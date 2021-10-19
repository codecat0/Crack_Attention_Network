"""
@File : crackattnet.py
@Author : CodeCat
@Time : 2021/9/6 上午9:58
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.my_att import SACAttention


class EncoderX2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderX2, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class EncoderX3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderX3, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Decoder, self).__init__()
        self.up_sample = nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=2, stride=2)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.att = SACAttention(in_channels=out_channels)

    def forward(self, x_prev, x):
        x = self.up_sample(x)
        h, w = x_prev.shape[2:]
        if x_prev.shape != x.shape:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        x_prev = self.att(x_prev)
        x = torch.cat([x_prev, x], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        return x


class MidLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MidLayer, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x



class CrackAttNet(nn.Module):
    def __init__(self, num_classes):
        super(CrackAttNet, self).__init__()

        self.down_sample1 = EncoderX2(in_channels=3, out_channels=64)
        self.down_sample2 = EncoderX2(in_channels=64, out_channels=128)
        self.down_sample3 = EncoderX3(in_channels=128, out_channels=256)
        self.down_sample4 = EncoderX3(in_channels=256, out_channels=512)
        self.down_sample5 = EncoderX3(in_channels=512, out_channels=512)

        self.mid = MidLayer(in_channels=512, out_channels=1024)

        self.up_sample5 = Decoder(in_channels=1024, hidden_channels=1024, out_channels=512)
        self.up_sample4 = Decoder(in_channels=1024, hidden_channels=512, out_channels=512)
        self.up_sample3 = Decoder(in_channels=512, hidden_channels=512, out_channels=256)
        self.up_sample2 = Decoder(in_channels=256, hidden_channels=256, out_channels=128)
        self.up_sample1 = Decoder(in_channels=128, hidden_channels=128, out_channels=64)

        self.classifier5 = nn.Conv2d(512, num_classes, 1)
        self.classifier4 = nn.Conv2d(512, num_classes, 1)
        self.classifier3 = nn.Conv2d(256, num_classes, 1)
        self.classifier2 = nn.Conv2d(128, num_classes, 1)
        self.classifier1 = nn.Conv2d(64, num_classes, 1)

        self.classifier = nn.Conv2d(5*num_classes, num_classes, 1)

    def forward(self, x):
        h, w = x.shape[2:]
        x1, x = self.down_sample1(x)
        x2, x = self.down_sample2(x)
        x3, x = self.down_sample3(x)
        x4, x = self.down_sample4(x)
        x5, x = self.down_sample5(x)

        x = self.mid(x)

        x = self.up_sample5(x5, x)
        out5 = self.classifier5(x)
        out5 = F.interpolate(out5, size=(h, w), mode='bilinear', align_corners=True)

        x = self.up_sample4(x4, x)
        out4 = self.classifier4(x)
        out4 = F.interpolate(out4, size=(h, w), mode='bilinear', align_corners=True)

        x = self.up_sample3(x3, x)
        out3 = self.classifier3(x)
        out3 = F.interpolate(out3, size=(h, w), mode='bilinear', align_corners=True)

        x = self.up_sample2(x2, x)
        out2 = self.classifier2(x)
        out2 = F.interpolate(out2, size=(h, w), mode='bilinear', align_corners=True)

        x = self.up_sample1(x1, x)
        out1 = self.classifier1(x)
        out1 = F.interpolate(out1, size=(h, w), mode='bilinear', align_corners=True)

        x = torch.cat([out5, out4, out3, out2, out1], dim=1)
        x = self.classifier(x)

        return x, out1, out2, out3, out4, out5


if __name__ == '__main__':
    input = torch.randn(1, 3, 448, 448)
    model = CrackAttNet(2)
    out, out_1, out_2, out_3, out_4, out_5 = model(input)
    print(out.shape)
    print(out_1.shape)
    print(out_2.shape)
    print(out_3.shape)
    print(out_4.shape)
    print(out_5.shape)

