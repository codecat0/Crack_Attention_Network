"""
@File : train_val_utils.py
@Author : CodeCat
@Time : 2021/7/27 上午10:45
"""
import datetime
import torch


def train_one_epoch(model, optimizer, criterion, metric, dataloader, opt, device, epoch):
    model.train()
    train_loss = 0
    train_acc = 0
    train_pos_acc = 0
    train_neg_acc = 0
    train_mean_acc = 0
    train_miou = 0
    train_fwiou = 0
    metric.reset()
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        out = out.permute(0, 2, 3, 1).contiguous().view(-1, opt.num_classes)
        labels = labels.view(-1)
        loss = criterion(out, labels.long())

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric.add_batch(out, labels)
        train_acc = metric.pixel_accuracy()
        train_neg_acc, train_pos_acc = metric.class_pixel_accuracy()
        train_mean_acc = metric.mean_pixel_accuracy()
        train_miou = metric.mean_intersection_over_union()
        train_fwiou = metric.frequency_weight_intersection_over_union()
        print(
            '[{}] Epoch is [{}/{}] mini-batch is [{}/{}], train_loss is {:.5f}, train_acc is {:.5f}, train_pos_acc is {:.5f}, train_neg_acc is {:.5f}, train_mean_acc is {:.5f}, train MIoU is {:.5f}, train FWIoU is {:.5f}'.format(
                str(datetime.datetime.now())[:19], epoch + 1, opt.epochs, i + 1, len(dataloader), loss, train_acc, train_pos_acc, train_neg_acc, train_mean_acc, train_miou, train_fwiou
            ))

    return train_loss/len(dataloader), train_acc, train_pos_acc, train_neg_acc, train_mean_acc, train_miou, train_fwiou


@torch.no_grad()
def evalute(model, dataloader, metric, criterion, device, opt, epoch):
    model.eval()

    eval_loss = 0
    eval_acc = 0
    eval_pos_acc = 0
    eval_neg_acc = 0
    eval_mean_acc = 0
    eval_miou = 0
    eval_fwiou = 0
    metric.reset()
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        out = out.permute(0, 2, 3, 1).contiguous().view(-1, opt.num_classes)
        labels = labels.view(-1).long()
        loss = criterion(out, labels)
        eval_loss += loss.item()

        metric.add_batch(out, labels)
        eval_acc = metric.pixel_accuracy()
        eval_neg_acc, eval_pos_acc = metric.class_pixel_accuracy()
        eval_mean_acc = metric.mean_pixel_accuracy()
        eval_miou = metric.mean_intersection_over_union()
        eval_fwiou = metric.frequency_weight_intersection_over_union()
        print(
            '[{}] Epoch is [{}/{}] mini-batch is [{}/{}], eval_loss is {:.5f}, eval_acc is {:.5f}, eval_pos_acc is {:.5f}, eval_neg_acc is {:.5f}, eval_mean_acc is {:.5f}, eval MIoU is {:.5f}, eval FWIoU is {:.5f}'.format(
                str(datetime.datetime.now())[:19], epoch + 1, opt.epochs, i + 1, len(dataloader), loss, eval_acc,
                eval_pos_acc, eval_neg_acc, eval_mean_acc, eval_miou, eval_fwiou
            ))

    return eval_loss/len(dataloader), eval_acc, eval_pos_acc, eval_neg_acc, eval_mean_acc, eval_miou, eval_fwiou