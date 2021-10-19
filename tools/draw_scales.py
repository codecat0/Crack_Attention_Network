"""
@File : draw_scales.py
@Author : CodeCat
@Time : 2021/10/19 上午10:17
"""
import numpy as np
import matplotlib.pyplot as plt

from src.tools.get_metrics import get_data_metric

import scikitplot as skplt


def plot_recall(model_name_list, y_true_list, y_probas_list):
    ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    i = 0
    for y_true, y_probas in zip(y_true_list, y_probas_list):
        y_true = np.array(y_true)
        y_probas = np.array(y_probas)

        classes = np.unique(y_true)
        probas = y_probas

        binarized_y_true = skplt.metrics.label_binarize(y_true, classes=classes)
        binarized_y_true = np.hstack(
            (1 - binarized_y_true, binarized_y_true)
        )

        average_precison = skplt.metrics.average_precision_score(binarized_y_true[:, 1], probas[:, 1])
        precison, recall, _ = skplt.metrics.precision_recall_curve(y_true, probas[:, 1], pos_label=classes[1])
        plt.plot(recall, precison, label='{} (F={:.3f})'.format(model_name_list[i], average_precison))
        i += 1
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.grid(linestyle='-.')
    plt.xlabel('Recall')
    plt.ylabel('Precison')
    plt.legend(loc='best')
    # plt.savefig('recall_scale.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    left_path = './tools/checkpoints/'
    right_path = '/last.pt'
    y_true_list = []
    y_probs_list = []
    model_name = 'Crack-Att Net'
    model_name_list = ['Crack-Att Net', 'Scale 1', 'Scale 2', 'Scale 3', 'Scale 4', 'Scale 5']
    for idx in range(len(model_name_list)):
        _, y_true, y_probs = get_data_metric(
            model_name=model_name,
            model_path=left_path + model_name + right_path,
            img_dir='../data/images_test',
            lab_dir='../data/masks_test',
            idx=idx
        )
        y_true_list.append(y_true)
        y_probs_list.append(y_probs)

    plot_recall(model_name_list, y_true_list, y_probs_list)