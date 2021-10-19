"""
@File : base_model.py
@Author : CodeCat
@Time : 2021/7/31 下午3:01
"""
import torch.nn as nn

from .fcn import FCN
from .unet import UNet
from .rcf import RCF
from .segnet import SegNet
from .deepcrack import DeepCrack
from .cracknet import CrackNet
from .crackattnet import CrackAttNet
from .crackcbamnet import CrackCbamNet



class BaseModel(nn.Module):
    def __init__(self, name, num_classes):
        super(BaseModel, self).__init__()
        if name == 'FCN':
            self.base = FCN(num_classes)
            self.multi_output = False
            self.multi_input = False
        elif name == 'U-Net':
            self.base = UNet(num_classes)
            self.multi_output = False
            self.multi_input = False
        elif name == 'RCF':
            self.base = RCF(num_classes)
            self.multi_output = True
            self.multi_input = False
        elif name == 'SegNet':
            self.base = SegNet(num_classes)
            self.multi_output = False
            self.multi_input = False
        elif name == 'DeepCrack':
            self.base = DeepCrack(num_classes)
            self.multi_output = True
            self.multi_input = False
        elif name == 'CrackNet':
            self.base = CrackNet(num_classes)
            self.multi_output = True
            self.multi_input = False
        elif name == 'Crack-Att Net':
            self.base = CrackAttNet(num_classes)
            self.multi_output = True
            self.multi_input = False
        elif name == 'Crack-Cbam Net':
            self.base = CrackCbamNet(num_classes)
            self.multi_output = True
            self.multi_input = False

    def forward(self, x):
        return self.base(x)


