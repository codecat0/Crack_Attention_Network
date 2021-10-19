"""
@File : ct.py
@Author : CodeCat
@Time : 2021/7/26 上午9:50
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Image_edge_detect:
    def __init__(self, img_gary):
        self.img = cv2.GaussianBlur(img_gary, (3, 3), 0)

    def get_robert(self):
        # Roberts算子
        kernel_x = np.array([[-1, 0], [0, 1]], dtype=np.int32)
        kernel_y = np.array([[0, -1], [1, 0]], dtype=np.int32)

        # 经过算子操作后会出现负值和大于225的值，故使用16位有符号数据类型
        x = cv2.filter2D(self.img, cv2.CV_16S, kernel_x)
        y = cv2.filter2D(self.img, cv2.CV_16S, kernel_y)

        # 转换回uint8，然后进行图像融合
        x = cv2.convertScaleAbs(x)
        y = cv2.convertScaleAbs(y)
        robert_img = cv2.addWeighted(x, 0.5, y, 0.5, 0)
        return robert_img


    def get_prewitt(self):
        # Prewiit 算子
        kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.int32)
        kernel_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.int32)

        x = cv2.filter2D(self.img, cv2.CV_16S, kernel_x)
        y = cv2.filter2D(self.img, cv2.CV_16S, kernel_y)

        x = cv2.convertScaleAbs(x)
        y = cv2.convertScaleAbs(y)
        prewiit_img = cv2.addWeighted(x, 0.5, y, 0.5, 0)
        return prewiit_img


    def get_laplacian(self):
        dst = cv2.Laplacian(self.img, cv2.CV_16S, ksize=3)
        laplacian_img = cv2.convertScaleAbs(dst)
        return laplacian_img


    def get_sobel(self):
        x = cv2.Sobel(self.img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(self.img, cv2.CV_16S, 0, 1)

        x = cv2.convertScaleAbs(x)
        y = cv2.convertScaleAbs(y)

        sobel_img = cv2.addWeighted(x, 0.5, y, 0.5, 0)
        return sobel_img

    def get_canny(self):
        canny_img = cv2.Canny(self.img, threshold1=50, threshold2=150)
        return canny_img




if __name__ == '__main__':
    img_path = '../DeepCrackData/test_img/11125-3.jpg'
    img = cv2.imread(filename=img_path)
    gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    titles = ['Origin Image', 'Gray Image', 'Gauss Blur', 'Robert',
              'Prewiit', 'Laplacian', 'Sobel', 'Canny']
    images = [img, gary]
    ied = Image_edge_detect(img_gary=gary)
    images.append(ied.img)
    images.append(ied.get_robert())
    images.append(ied.get_prewitt())
    images.append(ied.get_laplacian())
    images.append(ied.get_sobel())
    images.append(ied.get_canny())
    for i in range(len(titles)):
        plt.subplot(2, 4, i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()
