import torch as t
import os
import numpy as np
from torchvision import transforms
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from thop.utils import clever_format
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from train_detail import train_detail


train_opt = train_detail().parse()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

    return paths

def whitening(im): 
    batch_size, channel, h, w = im.shape
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    im = t.cat([(im[:,[0]]-mean[0])/std[0],
                    (im[:,[1]]-mean[1])/std[1],
                    (im[:,[2]]-mean[2])/std[2]], 1)
    return im


def l2_norm(x):
    norm = t.norm(x, p=2, dim=1, keepdim=True)
    x = t.div(x, norm)
    return x


def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0) 
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()  
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def eval(model, dataloader_valid, criterion):
    sum = 0
    val_loss_sum = 0
    val_top1_sum = 0
    model.eval()
    for ims, label in dataloader_valid:
        input_val = ims.cuda()
        target_val = label.cuda()
        output_val = model(input_val)  
        loss = criterion(output_val, target_val)

        top1_val = accuracy(output_val.data, target_val.data, topk=(1,))

        sum += 1 
        val_loss_sum += loss.data.cpu().numpy()
        val_top1_sum += top1_val[0]
    avg_loss = val_loss_sum / sum
    avg_top1 = val_top1_sum / sum

    return avg_loss, avg_top1





def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = t.randperm(batch_size).cuda()
    else:
        index = t.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

class GradualWarmupScheduler(_LRScheduler):

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch: #
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None): 
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)    




def select_sehcduler(lr_name, optimizer=None, Step_size = None , Multiplier=None, Total_epoch=None, Num_epochs=None):
    if lr_name == 'warmup':
        scheduler_cosine = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer,min_lr=0.00001)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=Multiplier, total_epoch=Total_epoch,
                                           after_scheduler=scheduler_cosine)
        return scheduler

    elif lr_name == 'steplr':
        scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=Step_size, gamma=0.01)

        return scheduler

    elif lr_name == 'cosin':
        scheduler = t.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=Total_epoch, eta_min=0.00001)


        return scheduler

    # return scheduler



def opcounter(model):

    from thop import profile
    input = t.randn(1, 3, 100, 100)
    flops, params = profile(model, inputs=(input, ))
    macs, params = clever_format([flops, params], "%.3f")

    print('total macs : {} '.format(macs))
    print('total params : {}'.format(params))

def split_data(X, Y, test_size=0.7, val_size=0.7):
  x_train, x_unlabel, y_train, y_unlabel = train_test_split(X, Y, test_size=test_size, random_state=42)
  x_val, x_unlabel, y_val, y_unlabel = train_test_split(x_unlabel, y_unlabel, test_size=val_size, random_state=42)
  return x_train, x_val, x_unlabel, x_unlabel, y_train, y_val, y_unlabel, y_unlabel



def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]
