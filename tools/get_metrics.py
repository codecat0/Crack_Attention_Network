"""
@File : get_metrics.py
@Author : CodeCat
@Time : 2021/10/11 下午9:36
"""
import os
import json
import collections
import cv2
import numpy as np


import torch
from src.model.base_model import BaseModel
from src.utils.metric import Metric
from src.dataset.custom_dataset import _preprocess_label, _preprocess_img



@torch.no_grad()
def get_data_metric(model_name, model_path, img_dir, lab_dir, idx=0):
    device = torch.device('cuda:0')
    model = BaseModel(model_name, 2)
    model.eval()
    model_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(model_dict['model'])
    model.to(device)
    imagepaths = sorted(os.listdir(img_dir))
    labelpaths = sorted(os.listdir(lab_dir))
    metric = Metric(num_classes=2, device=device)
    y_true_list = []
    y_probs_list = []
    for f, l in zip(imagepaths, labelpaths):
        image = cv2.imread(os.path.join(img_dir, f))
        label = cv2.imread(os.path.join(lab_dir, l), cv2.IMREAD_GRAYSCALE)

        img = _preprocess_img(image).to(device)
        label = _preprocess_label(label).to(device)
        x_input = img.unsqueeze(0)
        out = model(x_input)
        if model.multi_output:
            probs = out[idx]
        else:
            probs = out
        probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        label = label.view(-1)
        y_true_list.append(label.cpu().detach().numpy())
        y_probs_list.append(probs.softmax(1).cpu().detach().numpy())
        metric.add_batch(probs, label)
    precision = metric.get_precision().cpu().detach().numpy()
    recall = metric.get_recall().cpu().detach().numpy()
    f_score = metric.get_f_score().cpu().detach().numpy()
    miou = metric.mean_intersection_over_union().cpu().detach().numpy()
    pa = metric.pixel_accuracy().cpu().detach().numpy()
    mpa = metric.mean_pixel_accuracy().cpu().detach().numpy()
    fwiou = metric.frequency_weight_intersection_over_union().cpu().detach().numpy()
    metric_dict = {
        'Precision': float(precision),
        'Recall': float(recall),
        'F-score': float(f_score),
        'PA': float(pa),
        'MPA': float(mpa),
        'MIoU': float(miou),
        'FWIoU': float(fwiou)
    }
    y_true = np.array(y_true_list).reshape(-1)
    y_probs = np.array(y_probs_list).reshape(-1, 2)
    return metric_dict, y_true, y_probs




if __name__ == '__main__':
    left_path = './tools/checkpoints/'
    right_path = '/last.pt'
    export_path = '../export_data/'
    model_name_list = ['Crack-Att Net', 'DeepCrack', 'CrackNet', 'FCN', 'SegNet', 'RCF', 'U-Net']
    # model_name_list = ['Crack-Cbam Net']
    y_true_list = []
    y_probs_list = []
    models_metric_dict = collections.defaultdict(dict)
    for model_name in model_name_list:
        metric_dict, y_true, y_probs = get_data_metric(
            model_name=model_name,
            model_path=left_path + model_name + right_path,
            img_dir='../data/images',
            lab_dir='../data/masks'
        )
        models_metric_dict[model_name] = metric_dict
        y_true_list.append(y_true)
        y_probs_list.append(y_probs)

    with open(export_path + 'models_metric.json', 'w') as f:
        json.dump(models_metric_dict, f)
