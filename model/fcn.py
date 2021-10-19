"""
@File : fcn.py
@Author : CodeCat
@Time : 2021/7/26 下午8:54
"""
import torch
from torch import nn


from ..backbone.resnet import resnet34


class FCN(nn.Module):
    # https://arxiv.org/abs/1411.4038
    def __init__(self, pretrained=False, num_classes=2):
        super(FCN, self).__init__()
        backbone = resnet34()
        if pretrained:
            backbone.load_state_dict(torch.load('./backbone/weights/resnet34.pth'))

        self.stage1 = nn.Sequential(*list(backbone.children())[:-5])  # 1/4
        self.stage2 = backbone.layer2  # 1/8
        self.stage3 = backbone.layer3  # 1/16
        self.stage4 = backbone.layer4  # 1/32



        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.stage1(x)       # 1/4
        x2 = self.stage2(x1)      # 1/8
        x3 = self.stage3(x2)      # 1/16
        x4 = self.stage4(x3)      # 1/32

        score = self.upsample1(x4)  # 1/32 -> 1/16
        score = score + x3
        score = self.upsample2(score)   # 1/16 -> 1/8
        score = score + x2
        score = self.upsample3(score)   # 1/8 -> 1/4
        score = score + x1
        score = self.upsample4(score)   # 1/4 -> 1/1
        score = self.classifier(score)
        return score


if __name__ == '__main__':
    input = torch.randn(1, 3, 384, 544)
    model = FCN(pretrained=False, num_classes=2)
    output = model(input)
    print(output.shape)
