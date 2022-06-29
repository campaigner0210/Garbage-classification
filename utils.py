import torch
import shutil


def accuracy(output, target, topk=(1,)):

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)  # 求tensor中某个dim的前k大或者前k小的值以及对应的index。求一个样本被网络认为前k个最可能属于的类别
        pred = pred.t()  # 转tensor
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # .eq相应位置匹配

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):

    pre = str(state['valid_prec1']) + '_'
    torch.save(state, 'model/' + pre + filename)
    if is_best:
        shutil.copyfile('model/' + fold + filename, 'model/model_best/best_' + filename + '{}'.format(str(time.strftime('%Y-%m-%d %H:%M:%S').replace(' ', '_')))) # 复制文件到另一个文件夹中


class AverageMeter(object):

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
