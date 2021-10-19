"""
@File : data_augment.py
@Author : CodeCat
@Time : 2021/7/27 下午4:35
"""
import cv2
import numpy as np


class AugCompose(object):
    def __init__(self, transforms=None):
        self.transforms = transforms

    def __call__(self, img, label):
        if self.transforms is not None:
            for op, prob in self.transforms:
                if np.random.rand() <= prob:
                    img, label = op(img, label)

        return img, label


def RandomFlip(img, label, FLIP_LEFT_RIGHT=True, FLIP_TOP_BOTTOM=True):
    """
    Flip Image
    """
    # horizontal flip
    if FLIP_LEFT_RIGHT and np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        label = cv2.flip(label, 1)

    # vertical flip
    if FLIP_TOP_BOTTOM and np.random.rand() < 0.5:
        img = cv2.flip(img, 0)
        label = cv2.flip(label, 0)


    return img, label


def RandomBlur(img, label):
    """
    Image Filter
    """
    r = 5

    if np.random.rand() < 0.2:
        return cv2.GaussianBlur(img, (r, r), 0), label

    if np.random.rand() < 0.15:
        return cv2.blur(img, (r, r)), label

    if np.random.rand() < 0.1:
        return cv2.medianBlur(img, r), label

    return img, label


def RandomColorJitter(img, label, brightness=32, contrast=0.5, saturation=0.5, hue=0.1, prob=0.5):
    """
    Change Image ColorJitter
    """

    if brightness != 0 and np.random.rand() > prob:
        img = _Brightness(img, delta=brightness)

    if contrast != 0 and np.random.rand() > prob:
        img = _Contrast(img, var=contrast)

    if saturation != 0 and np.random.rand() > prob:
        img = _Saturation(img, var=saturation)

    if hue != 0 and np.random.rand() > prob:
        img = _Hue(img, var=hue)

    return img, label


def t_random(min, max):
    return min + (max - min) * np.random.rand()


def t_randint(min, max):
    return np.random.randint(min, max)


def _Brightness(img, delta=32):
    """
    Change Image Brightness
    """
    img = img.astype(np.float32) + t_random(-delta, delta)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def _Contrast(img, var=0.3):
    """
    Change Image Contrast
    """
    gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()
    alpha = 1.0 + t_random(-var, var)
    img = alpha * img.astype(np.float32) + (1 - alpha) * gs
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def _Hue(img, var=0.05):
    """
    Change Image Hue
    """
    var = t_random(-var, var)
    to_HSV, form_HSV = [
        (cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB),
        (cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR)
    ][t_randint(0, 2)]
    hsv = cv2.cvtColor(img, to_HSV).astype(np.float32)

    hue = hsv[:, :, 0] / 179. + var
    hue = hue - np.floor(hue)
    hsv[:, :, 0] = hue * 179.

    img = cv2.cvtColor(hsv.astype('uint8'), form_HSV)
    return img


def _Saturation(img, var=0.3):
    """
    Change Image Saturation
    """

    gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gs = np.expand_dims(gs, axis=2)
    alpha = 1.0 + t_random(-var, var)
    img = alpha * img.astype(np.float32) + (1 - alpha) * gs.astype(np.float32)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

