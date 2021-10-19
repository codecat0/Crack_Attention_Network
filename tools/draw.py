"""
@File : draw.py
@Author : CodeCat
@Time : 2021/8/20 下午7:31
"""
import collections
import numpy as np
import matplotlib.pyplot as plt

from src.tools.get_metrics import get_data_metric

import scikitplot as skplt





def plot_roc(model_name_list, y_true_list, y_probas_list):
    ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    i = 0
    for y_true, y_probas in zip(y_true_list, y_probas_list):
        y_true = np.array(y_true)
        y_probas = np.array(y_probas)

        classes = np.unique(y_true)
        probas = y_probas

        fpr_dict = dict()
        tpr_dict = dict()

        fpr_dict[1], tpr_dict[1], _ = skplt.metrics.roc_curve(y_true, probas[:, 1], pos_label=classes[1])
        roc_auc = skplt.metrics.auc(fpr_dict[1], tpr_dict[1])
        plt.plot(fpr_dict[1], tpr_dict[1], label='{} (auc={:.3f})'.format(model_name_list[i], roc_auc))
        i += 1
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.grid(linestyle='-.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.savefig('roc.png', dpi=600)
    plt.show()


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
    model_name_list = ['Crack-Att Net', 'DeepCrack', 'CrackNet', 'FCN', 'SegNet', 'U-Net', 'RCF']
    y_true_list = []
    y_probs_list = []
    # models_metric_dict = collections.defaultdict(dict)
    for model_name in model_name_list:
        _, y_true, y_probs = get_data_metric(
            model_name=model_name,
            model_path=left_path + model_name + right_path,
            img_dir='../data/images',
            lab_dir='../data/masks'
        )
        # models_metric_dict[model_name] = metric_dict
        y_true_list.append(y_true)
        y_probs_list.append(y_probs)



    plot_recall(model_name_list, y_true_list, y_probs_list)

