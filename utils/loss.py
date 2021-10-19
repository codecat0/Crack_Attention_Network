"""
@File : loss.py
@Author : CodeCat
@Time : 2021/7/28 上午9:57
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# 二分类
class FocalLoss(nn.Module):

    def __init__(self, gamma=2, alpha=0.5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        # input:size is M*2. M　is the batch　number
        # target:size is M.
        pt = torch.softmax(input, dim=1)
        p = pt[:, 1]
        loss = -self.alpha * (1 - p) ** self.gamma * (target * torch.log(p)) - (1 - self.alpha) * p ** self.gamma * ((1 - target) * torch.log(1 - p))
        return loss.mean()


class CEWeightLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super(CEWeightLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        # input:size is M*2. M　is the batch　number
        # target:size is M.
        weight = torch.sum(target) / target.numel()
        weights = torch.Tensor([weight, 1-weight]).to(input.device)
        return F.cross_entropy(input, target, weight=weights, reduction=self.reduction)


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, input, target):
        # input:size is M*2. M　is the batch　number
        # target:size is M.
        neg = 1 - target
        pos = target
        target = torch.stack((neg, pos), dim=1)
        intersection = torch.sum(input * target)
        union = torch.sum(input) + torch.sum(target) + self.eps
        loss = 1 - 2.0 * intersection / union
        return loss
