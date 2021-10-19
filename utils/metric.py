"""
@File : metric.py
@Author : CodeCat
@Time : 2021/7/27 上午9:44
"""
import torch
import numpy as np
import scikitplot as skplt


class Metric(object):
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.confusion_matrix = torch.zeros((num_classes, num_classes)).to(device)

    def gen_confusion_matrix(self, img_predict, img_label):
        mask = (img_label >= 0) & (img_label < self.num_classes)
        label = self.num_classes * img_label[mask] + img_predict[mask]
        count = torch.bincount(label, minlength=self.num_classes*2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def add_batch(self, img_predict, img_label):
        img_predict = torch.argmax(img_predict, dim=1)
        # classes = torch.unique(img_label)
        # img_predict = img_predict * classes[1]

        assert img_predict.shape == img_label.shape

        self.confusion_matrix += self.gen_confusion_matrix(img_predict, img_label)

    def get_precision(self):
        class_precision = torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(0)
        precision = class_precision[1]
        return precision

    def get_recall(self):
        class_recall = torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(1)
        recall = class_recall[1]
        return recall

    def get_f_score(self):
        return 2 * self.get_precision() * self.get_recall() / (self.get_precision() + self.get_recall())

    def pixel_accuracy(self):
        """
        获取像素准确率
        PA = (TP + TN) / (TP + TN + FP + FN)
        :return:
        """
        acc = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def class_pixel_accuracy(self):
        """
        获取类别像素准确率
        CPA = [
            class_1 : TN / (TN + FN)
            class_2 : TP / (TP + FP)
            ...
        ]
        :return:
        """
        class_acc = torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(0)
        return class_acc

    def mean_pixel_accuracy(self):
        """
        获取类别平均像素准确率
        """
        class_acc = self.class_pixel_accuracy()
        mean_acc = torch.mean(class_acc)
        return mean_acc

    def mean_intersection_over_union(self):
        """
        获取平均交并比
        MIoU = [TP / (TP + FP + FN) + TN / (TN + FN + FP)] / 2
        """
        intersection = torch.diag(self.confusion_matrix)
        union = torch.sum(self.confusion_matrix, 1) + torch.sum(self.confusion_matrix, 0) - torch.diag(self.confusion_matrix)
        IoU = intersection / union
        mIoU = torch.mean(IoU)
        return mIoU

    def frequency_weight_intersection_over_union(self):
        """
        频权交并比
        """
        freq = torch.sum(self.confusion_matrix, 1) / torch.sum(self.confusion_matrix)
        iu = torch.diag(self.confusion_matrix) / (
            torch.sum(self.confusion_matrix, 1) + torch.sum(self.confusion_matrix, 0) - torch.diag(self.confusion_matrix)
        )
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes)).to(self.device)