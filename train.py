from dataset import Garbage_Loader
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch
import time
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
from utils import accuracy, save_checkpoint, AverageMeter
from progress.bar import Bar


def train(train_loader, model, criterion, optimizer, epoch, writer):

    batch_time = AverageMeter()  # 训练一个batch
    data_time = AverageMeter()  # 训练一次数据集
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    bar = Bar('Epoch[{}]Train'.format(epoch), max=len(train_loader))

    # switch to train mode
    model.train()

    end = time.time()
    for batch_idx, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        [prec1, prec5] = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    writer.add_scalar('train_loss', losses.val, global_step=epoch)


def validate(val_loader, model, criterion, epoch, writer, phase="VAL"):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    bar = Bar('Epoch[{}]Val'.format(epoch)', max=len(val_loader))

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            [prec1, prec5] = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = '({batch}/{size}) Time: {time:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=i + 1,
                size=len(val_loader),
                time=batch_time.val,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
        bar.finish()
        print(' * {} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(phase, top1=top1, top5=top5))
    writer.add_scalar('valid_loss', losses.val, global_step=epoch)
    return top1.avg, top5.avg


if __name__ == "__main__":
    # -------------------------------------------- step 1/4 : 加载数据 ---------------------------
    train_dir_list = 'train.txt'
    valid_dir_list = 'val.txt'
    batch_size = 64
    epochs = 10
    num_classes = 214 # 垃圾共214类
    train_data = Garbage_Loader(train_dir_list, train_flag=True)
    valid_data = Garbage_Loader(valid_dir_list, train_flag=False)
    train_loader = DataLoader(dataset=train_data, num_workers=2, pin_memory=True, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, num_workers=2, pin_memory=True, batch_size=batch_size)
    train_data_size = len(train_data)
    print('训练集数量：%d' % train_data_size)
    valid_data_size = len(valid_data)
    print('验证集数量：%d' % valid_data_size)
    # ------------------------------------ step 2/4 : 定义网络 ------------------------------------
    model = models.resnet101(pretrained=True)
    fc_inputs = model.fc.in_features
    model.fc = nn.Linear(fc_inputs, num_classes)
    model = model.cuda()
    # ------------------------------------ step 3/4 : 定义损失函数和优化器等 -------------------------
    learn_rate = 1e-4
    lr_stepsize = 20
    weight_decay = 1e-3
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_stepsize, gamma=0.1)

    writer = SummaryWriter('SummaryWriter/resnet101')
    # ------------------------------------ step 4/4 : 训练 -----------------------------------------
    best_prec1 = 0
    for epoch in range(epochs):
        train(train_loader, model, criterion, optimizer, epoch, writer)
        scheduler.step()
        # 在验证集上测试效果
        valid_prec1, valid_prec5 = validate(valid_loader, model, criterion, epoch, writer, phase="VAL")
        is_best = valid_prec1 > best_prec1
        best_prec1 = max(valid_prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'resnet101',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best,
            filename='checkpoint_resnet101.pth.tar')
    writer.close()
