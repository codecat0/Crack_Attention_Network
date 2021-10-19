"""
@File : unet.py
@Author : CodeCat
@Time : 2021/7/27 下午9:14
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_prev, x):
        x = self.up_sample(x)
        x_shape = x.shape[2:]
        x_prev_shape = x.shape[2:]
        h_diff = x_prev_shape[0] - x_shape[0]
        w_diff = x_prev_shape[1] - x_shape[1]
        # padding
        x_tmp = torch.zeros(x_prev.shape).to(x.device)
        x_tmp[:, :, h_diff//2: h_diff+x_shape[0], w_diff//2: x_shape[1]] = x
        x = torch.cat([x_prev, x_tmp], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        return x



class UNet(nn.Module):
    # https://arxiv.org/abs/1505.04597
    def __init__(self, num_classes=2):
        super(UNet, self).__init__()

        self.down_sample1 = Encoder(in_channels=3, out_channels=64)
        self.down_sample2 = Encoder(in_channels=64, out_channels=128)
        self.down_sample3 = Encoder(in_channels=128, out_channels=256)
        self.down_sample4 = Encoder(in_channels=256, out_channels=512)

        self.mid1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, bias=False),
            nn.ReLU(inplace=True)
        )
        self.mid2 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, bias=False),
            nn.ReLU(inplace=True)
        )

        self.up_sample1 = Decoder(in_channels=1024, out_channels=512)
        self.up_sample2 = Decoder(in_channels=512, out_channels=256)
        self.up_sample3 = Decoder(in_channels=256, out_channels=128)
        self.up_sample4 = Decoder(in_channels=128, out_channels=64)

        self.classifier = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x1, x = self.down_sample1(x)
        x2, x = self.down_sample2(x)
        x3, x = self.down_sample3(x)
        x4, x = self.down_sample4(x)

        x = self.mid1(x)
        x = self.mid2(x)

        x = self.up_sample1(x4, x)
        x = self.up_sample2(x3, x)
        x = self.up_sample3(x2, x)
        x = self.up_sample4(x1, x)

        x = self.classifier(x)
        return x



if __name__ == '__main__':
    input = torch.rand(1, 3, 384, 384)
    model = UNet(2)
    out = model(input)
    print(out.shape)



