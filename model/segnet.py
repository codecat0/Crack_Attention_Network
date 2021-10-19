"""
@File : segnet.py
@Author : CodeCat
@Time : 2021/8/3 下午7:07
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()

        batchNorm_momentum = 0.1

        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),
        )

        self.encode2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),
        )

        self.encode3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),
        )

        self.encode4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),
        )

        self.encode5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        idx = []

        x = self.encode1(x)
        x, id1 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2, return_indices=True)
        idx.append(id1)

        x = self.encode2(x)
        x, id2 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2, return_indices=True)
        idx.append(id2)

        x = self.encode3(x)
        x, id3 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2, return_indices=True)
        idx.append(id3)

        x = self.encode4(x)
        x, id4 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2, return_indices=True)
        idx.append(id4)

        x = self.encode5(x)
        x, id5 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2, return_indices=True)
        idx.append(id5)

        return x, idx


class Decoder(nn.Module):
    def __init__(self, out_channels):
        super(Decoder, self).__init__()

        batchNorm_momentum = 0.1

        self.decode1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True)
        )

        self.decode2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True)
        )

        self.decode3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True)
        )

        self.decode4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True)
        )

        self.decode5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, idx):
        x = F.max_unpool2d(x, idx[4], kernel_size=2, stride=2)

        x = self.decode1(x)
        x = F.max_unpool2d(x, idx[3], kernel_size=2, stride=2)

        x = self.decode2(x)
        x = F.max_unpool2d(x, idx[2], kernel_size=2, stride=2)

        x = self.decode3(x)
        x = F.max_unpool2d(x, idx[1], kernel_size=2, stride=2)

        x = self.decode4(x)
        x = F.max_unpool2d(x, idx[0], kernel_size=2, stride=2)

        x = self.decode5(x)

        return x


class SegNet(nn.Module):
    # https://arxiv.org/abs/1511.00561
    def __init__(self, num_classes):
        super(SegNet, self).__init__()

        self.encode = Encoder(in_channels=3)
        self.decode = Decoder(out_channels=num_classes)

    def forward(self, x):
        x, idx = self.encode(x)
        x = self.decode(x, idx)
        return x


if __name__ == '__main__':
    input = torch.randn(1, 3, 384, 544)
    model = SegNet(num_classes=2)
    output = model(input)
    print(output.shape)