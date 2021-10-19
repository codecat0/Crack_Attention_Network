"""
@File : train_val_mutil_utils.py
@Author : CodeCat
@Time : 2021/7/31 上午11:33
"""
import datetime
import torch


def train_one_epoch_multi(model, optimizer, metric, criterion, dataloader, opt, device, epoch):
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

        output, fuse5, fuse4, fuse3, fuse2, fuse1 = model(images)

        output = output.permute(0, 2, 3, 1).contiguous().view(-1, opt.num_classes)
        fuse5 = fuse5.permute(0, 2, 3, 1).contiguous().view(-1, opt.num_classes)
        fuse4 = fuse4.permute(0, 2, 3, 1).contiguous().view(-1, opt.num_classes)
        fuse3 = fuse3.permute(0, 2, 3, 1).contiguous().view(-1, opt.num_classes)
        fuse2 = fuse2.permute(0, 2, 3, 1).contiguous().view(-1, opt.num_classes)
        fuse1 = fuse1.permute(0, 2, 3, 1).contiguous().view(-1, opt.num_classes)

        labels = labels.view(-1).long()

        loss_output = criterion(output, labels)
        loss_fuse5 = criterion(fuse5, labels)
        loss_fuse4 = criterion(fuse4, labels)
        loss_fuse3 = criterion(fuse3, labels)
        loss_fuse2 = criterion(fuse2, labels)
        loss_fuse1 = criterion(fuse1, labels)

        loss = loss_output + loss_fuse5 + loss_fuse4 + loss_fuse3 + loss_fuse2 + loss_fuse1
        # case1
        # loss = loss_output + loss_fuse5 + 1/3 * loss_fuse4 + 1/9 * loss_fuse3 + 1/27 * loss_fuse2 + 1/81 * loss_fuse1
        # case2
        # loss = loss_output + loss_fuse5 + 1/2 * loss_fuse4 + 1/4 * loss_fuse3 + 1/8 * loss_fuse2 + 1/16 * loss_fuse1
        # case3
        # loss = loss_output + loss_fuse5 + loss_fuse4 + 1/2 * loss_fuse3 + 1/4 * loss_fuse2 + 1/8 * loss_fuse1
        # case4
        # loss = loss_output + loss_fuse5 + loss_fuse4 + 1/3 * loss_fuse3 + 1/9 * loss_fuse2 + 1/27 * loss_fuse1
        # case6
        # loss = loss_output + loss_fuse5 + loss_fuse4 + 3 * loss_fuse3 + 9 * loss_fuse2 + 27 * loss_fuse1
        # case7
        # loss = loss_output + loss_fuse5 + loss_fuse4 + 2 * loss_fuse3 + 4 * loss_fuse2 + 8 * loss_fuse1
        # case8
        # loss = loss_output + loss_fuse5 + 2 * loss_fuse4 + 4 * loss_fuse3 + 8 * loss_fuse2 + 16 * loss_fuse1

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric.add_batch(output, labels)
        train_acc = metric.pixel_accuracy()
        train_neg_acc, train_pos_acc = metric.class_pixel_accuracy()
        train_mean_acc = metric.mean_pixel_accuracy()
        train_miou = metric.mean_intersection_over_union()
        train_fwiou = metric.frequency_weight_intersection_over_union()
        print(
            '[{}] Epoch is [{}/{}] mini-batch is [{}/{}], train_loss is {:.5f}, train_acc is {:.5f}, train_pos_acc is {:.5f}, train_neg_acc is {:.5f}, train_mean_acc is {:.5f}, train MIoU is {:.5f}, train FWIoU is {:.5f}'.format(
                str(datetime.datetime.now())[:19], epoch + 1, opt.epochs, i + 1, len(dataloader), loss, train_acc,
                train_pos_acc, train_neg_acc, train_mean_acc, train_miou, train_fwiou
            ))

    return train_loss / len(
        dataloader), train_acc, train_pos_acc, train_neg_acc, train_mean_acc, train_miou, train_fwiou


@torch.no_grad()
def evalute_multi(model, dataloader, metric, criterion, device, opt, epoch):
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
        output, fuse5, fuse4, fuse3, fuse2, fuse1 = model(images)

        output = output.permute(0, 2, 3, 1).contiguous().view(-1, opt.num_classes)
        fuse5 = fuse5.permute(0, 2, 3, 1).contiguous().view(-1, opt.num_classes)
        fuse4 = fuse4.permute(0, 2, 3, 1).contiguous().view(-1, opt.num_classes)
        fuse3 = fuse3.permute(0, 2, 3, 1).contiguous().view(-1, opt.num_classes)
        fuse2 = fuse2.permute(0, 2, 3, 1).contiguous().view(-1, opt.num_classes)
        fuse1 = fuse1.permute(0, 2, 3, 1).contiguous().view(-1, opt.num_classes)

        labels = labels.view(-1).long()

        loss_output = criterion(output, labels)
        loss_fuse5 = criterion(fuse5, labels)
        loss_fuse4 = criterion(fuse4, labels)
        loss_fuse3 = criterion(fuse3, labels)
        loss_fuse2 = criterion(fuse2, labels)
        loss_fuse1 = criterion(fuse1, labels)

        loss = loss_output + loss_fuse5 + loss_fuse4 + loss_fuse3 + loss_fuse2 + loss_fuse1

        eval_loss += loss.item()

        metric.add_batch(output, labels)
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

    return eval_loss / len(dataloader), eval_acc, eval_pos_acc, eval_neg_acc, eval_mean_acc, eval_miou, eval_fwiou