"""
@File : custom_dataset.py
@Author : CodeCat
@Time : 2021/7/26 下午7:15
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt



class MyDataset(Dataset):
    def __init__(self, images_path='', labels_path='', transforms=None):
        self.transforms = transforms
        self.imgaes_list = []
        for item in sorted(os.listdir(images_path)):
            self.imgaes_list.append(os.path.join(images_path, item))
        self.labels_list = []
        for item in sorted(os.listdir(labels_path)):
            self.labels_list.append(os.path.join(labels_path, item))

    def __getitem__(self, item):
        image_path = self.imgaes_list[item]
        label_path = self.labels_list[item]

        img = cv2.imread(image_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if self.transforms is not None:
            img, lab = self.transforms(img, label)

        img = _preprocess_img(img)
        label = _preprocess_label(label)

        return img, label

    def __len__(self):
        return len(self.imgaes_list)


def _preprocess_img(cvImage):
    """
    :param cvImage: numpy HWC BGR 0~255
    :return: tensor img CHW BGR float32 cpu 0~1
    """

    cvImage = cvImage.transpose(2, 0, 1).astype(np.float32) / 255

    return torch.from_numpy(cvImage)


def _preprocess_label(cvImage):
    """
    :param cvImage: numpy 0(backgroud) or 255(crack pixel)
    :return: tensor 0 or 1 float32
    """

    cvImage[cvImage <= 128] = 0
    cvImage[cvImage > 128] = 1

    return torch.from_numpy(cvImage)



if __name__ == '__main__':
    dataset = MyDataset(images_path='../../CrackForest-dataset/image/', labels_path='../../CrackForest-dataset/groundTruthPngImg/')
    image, label = dataset[0]
    print(label)
    plt.imshow(label)
    plt.show()
    print(label[label != 0])


