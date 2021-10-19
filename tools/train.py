"""
@File : train.py
@Author : CodeCat
@Time : 2021/7/31 下午2:37
"""
import os
import math
import datetime
import argparse

import torch
from torch import optim
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset.custom_dataset import MyDataset
from src.dataset.data_augment import AugCompose, RandomFlip, RandomBlur, RandomColorJitter

from src.model.base_model import BaseModel

from src.utils.train_val_utils import train_one_epoch, evalute
from src.utils.train_val_multi_utils import train_one_epoch_multi, evalute_multi
from src.utils.loss import FocalLoss, CEWeightLoss, DiceLoss
from src.utils.metric import Metric


def main(opt):
    # Load DataSet
    nw = min([os.cpu_count(), opt.batch_size if opt.batch_size > 1 else 0, 8])

    data_transforms = AugCompose(
        [
            # (RandomFlip, 0.5),
            (RandomColorJitter, 0.5),
            (RandomBlur, 0.2),
        ]
    )

    train_dataset = MyDataset(images_path=opt.train_image_path, labels_path=opt.train_label_path, transforms=data_transforms)
    val_dataset = MyDataset(images_path=opt.val_image_path, labels_path=opt.val_image_path, transforms=None)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=nw,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=nw,
        drop_last=True,
        pin_memory=True,
    )

    # Definee loss
    if opt.use_focal_loss:
        criterion = FocalLoss()
    elif opt.use_dice_loss:
        criterion = DiceLoss()
    elif opt.use_weight_loss:
        criterion = CEWeightLoss()
    else:
        criterion = nn.CrossEntropyLoss()
        # criterion = CEWeightLoss()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(opt.log_dir):
        os.mkdir(opt.log_dir)
    log_dir = os.path.join(opt.log_dir, opt.model_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # 删除之前的日志
    for file in os.listdir(log_dir):
        file_path = os.path.join(log_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    tb_writer = SummaryWriter(log_dir=log_dir)

    # Load model
    model = BaseModel(name=opt.model_name, num_classes=opt.num_classes)
    model.to(device)


    # Optimizer
    if opt.use_adam:
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=5E-5)
    else:
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)

    lf = lambda x: ((1 + math.cos(x * math.pi / opt.epochs)) / 2) * (1 - opt.lrf) + opt.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # PreTrained
    start_epoch = 0
    if opt.weights != '':
        checkponits = torch.load(opt.weights, map_location=device)
        model.load_state_dict(checkponits['model'], strict=False)
        optimizer.load_state_dict(checkponits['optimizer'])
        start_epoch = checkponits['epoch'] + 1

    # Freeze
    if opt.freeze_layers:
        for name, parameter in model.named_parameters():
            if 'stage' in name:
                parameter.requires_grad_(False)

    print("[{}] \nUsing {} {} device for training, \ntrain_dataset is {}, val_dataset is {}, \nloss_function is {}, \noptimizer is {}".format(
        str(datetime.datetime.now())[:19], device, torch.cuda.get_device_name(device), len(train_dataset),
        len(val_dataset), criterion, optimizer))

    # metric
    metirc = Metric(num_classes=opt.num_classes, device=device)

    best_metric = 0
    for epoch in range(start_epoch, opt.epochs):
        # Train
        if model.multi_input or model.multi_output:
            train_loss, train_acc, train_pos_acc, train_neg_acc, train_mean_acc, train_miou, train_fwiou = train_one_epoch_multi(
                model=model,
                optimizer=optimizer,
                metric=metirc,
                criterion=criterion,
                dataloader=train_loader,
                opt=opt,
                device=device,
                epoch=epoch
            )
        else:
            train_loss, train_acc, train_pos_acc, train_neg_acc, train_mean_acc, train_miou, train_fwiou = train_one_epoch(
                model=model,
                optimizer=optimizer,
                metric=metirc,
                criterion=criterion,
                dataloader=train_loader,
                opt=opt,
                device=device,
                epoch=epoch
            )
        scheduler.step()
        # Eval
        if model.multi_output:
            eval_loss, eval_acc, eval_pos_acc, eval_neg_acc, eval_mean_acc, eval_miou, eval_fwiou = evalute_multi(
                model=model,
                criterion=criterion,
                metric=metirc,
                dataloader=val_loader,
                opt=opt,
                device=device,
                epoch=epoch
            )
        else:
            eval_loss, eval_acc, eval_pos_acc, eval_neg_acc, eval_mean_acc, eval_miou, eval_fwiou = evalute(
                model=model,
                criterion=criterion,
                metric=metirc,
                dataloader=val_loader,
                opt=opt,
                device=device,
                epoch=epoch
            )

        print('[{}] Epoch is [{}/{}], train_loss is {:.5f}, train_acc is {:.5f}, train_pos_acc is {:5f}, train_neg_acc is {:5f}, eval_loss is {:.5f}, eval_acc is {:.5f}, eval_pos_acc is {:.5f}, eval_neg_acc is {:.5f}'.format(
            str(datetime.datetime.now())[:19], epoch+1, opt.epochs, train_loss, train_acc, train_pos_acc, train_neg_acc, eval_loss, eval_acc, eval_pos_acc, eval_neg_acc
        ))
        # Save Best model
        if eval_pos_acc > best_metric:
            best_metric = eval_pos_acc
            if not os.path.exists(opt.save_path):
                os.mkdir(opt.save_path)

            state_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            best_model_path = os.path.join(opt.save_path, opt.model_name)
            if not os.path.exists(best_model_path):
                os.mkdir(best_model_path)
            torch.save(state_dict, os.path.join(best_model_path, 'best.pt'))
        # Save Last model
        if not os.path.exists(opt.save_path):
            os.mkdir(opt.save_path)
        state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        last_model_path = os.path.join(opt.save_path, opt.model_name)
        if not os.path.exists(last_model_path):
            os.mkdir(last_model_path)
        torch.save(state_dict, os.path.join(last_model_path, 'last.pt'))

        # Tensorboard
        tags = ['Loss', 'Pixel Accuracy', 'Mean Piexel Accuracy', 'MIoU', 'FWIoU']
        values = [train_loss, train_acc, train_mean_acc, train_miou, train_fwiou]
        for i in range(len(tags)):
            tb_writer.add_scalar(tags[i], values[i], epoch)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--train_image_path', type=str, default='../data/images_train')
    parser.add_argument('--train_label_path', type=str, default='../data/masks_train')
    parser.add_argument('--val_image_path', type=str, default='../data/images_test')
    parser.add_argument('--val_label_path', type=str, default='../data/masks_test')
    parser.add_argument('--batch_size', type=int, default=4)

    # model
    parser.add_argument('--model_name', type=str, default='FCN')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--freeze_layers', type=bool, default=False)

    # optimizer
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--use_adam', type=bool, default=False)

    # train
    parser.add_argument('--use_focal_loss', type=bool, default=False)
    parser.add_argument('--use_dice_loss', type=bool, default=False)
    parser.add_argument('--use_weight_loss', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save_path', type=str, default='./tools/checkpoints')

    # tensorborad
    parser.add_argument('--log_dir', type=str, default='./tools/runs')

    opt = parser.parse_args()
    print(opt)
    main(opt)
