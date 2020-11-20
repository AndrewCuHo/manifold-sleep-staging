import os
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import numpy as np
import pickle
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from common.misc import mkdir_p
from sklearn.preprocessing import OneHotEncoder
from loss import *
from model.utils import select_network
from train_detail import train_detail
from utils import accuracy, select_sehcduler, opcounter, split_data, get_spectrogram, to_categorical
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import scipy
from itertools import cycle
import itertools
train_opt = train_detail().parse()
train_path = train_opt.train_path
Num_classes = train_opt.num_classes
Size_height = train_opt.input_height
Size_weight = train_opt.input_weight
Model = train_opt.model
Checkpoint = train_opt.checkpoints
Resume = train_opt.resume
Loss = train_opt.loss
Num_epochs = train_opt.num_epochs
Batch_size = train_opt.batch_size
Freeze = train_opt.freeze
Init_lr = train_opt.init_lr
Lr_scheduler = train_opt.lr_scheduler
Step_size = train_opt.step_size
Multiplier = train_opt.multiplier
Total_epoch = train_opt.total_epoch
Alpha = train_opt.alpha
Gamma = train_opt.gamma
Re = train_opt.re
ManualSeed = train_opt.manualSeed
torch.manual_seed(ManualSeed)
torch.cuda.manual_seed_all(ManualSeed)
np.random.seed(ManualSeed)
random.seed(ManualSeed)
torch.backends.cudnn.deterministic = True
OutpuDir = train_opt.out
UnP = train_opt.UnlabeledPercent / 100
PThreshold = train_opt.Distrib_Threshold
Un_lamda = train_opt.Balance_loss
IF_GPU = train_opt.IF_GPU
IF_TRAIN = train_opt.IF_TRAIN
SelectData = train_opt.SelectData
def train():
    begin_time = time.time()
    model = select_network(Model, Num_classes)
    opcounter(model)
    if IF_GPU:
        model.cuda()
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        model.cpu()
    Normal_transform = transforms.Compose([
        transforms.ToPILImage(),
    X = pickle.load()
    del X[0]
    Normal_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((train_opt.crop_size, train_opt.crop_size))])
    Strong_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize((train_opt.crop_size, train_opt.crop_size))])
    img_trans = []
    for idx in X:
        img_tran = Normal_transform(idx)
        img_tran = np.asarray(img_tran, dtype="float32")
        img_trans.append(img_tran)
    Y = pickle.load()
    x_train, x_val, x_unlabel_weak, x_unlabel_strong, y_train, y_val, y_unlabel_weak, y_unlabel_strong = split_data(X,                                                                                                                   Y
    unlabe_strong = np.array(x_unlabel_strong, dtype=np.uint8)
    img_trans = []
    for idx in unlabe_strong:
        img_tran = Strong_transform(idx)
        img_tran = np.asarray(img_tran, dtype="float32")
        img_trans.append(img_tran)
    x_unlabel_strong = np.array(img_trans, dtype="float32")
    print('total data number : ', len(X))
    print('train number : ', len(x_train))
    print('unlabeled weak number : ', len(x_unlabel_weak))
    print('unlabeled strong number : ', len(x_unlabel_strong))
    print('validation number : ', len(x_val))
    X_train_valid = torch.from_numpy(x_train).float()
    y_train_valid = torch.from_numpy(y_train).long()
    dst_train = torch.utils.data.TensorDataset(X_train_valid, y_train_valid)
    X_train_valid = torch.from_numpy(x_unlabel_weak).float()
    y_train_valid = torch.from_numpy(y_unlabel_weak).long()
    dst_weak_unlabeltrain = torch.utils.data.TensorDataset(X_train_valid, y_train_valid)
    X_train_valid = torch.from_numpy(x_unlabel_strong).float()
    y_train_valid = torch.from_numpy(y_unlabel_strong).long()
    dst_strong_unlabeltrain = torch.utils.data.TensorDataset(X_train_valid, y_train_valid)
    X_train_valid = torch.from_numpy(x_val).float()
    y_train_valid = torch.from_numpy(y_val).long()
    dst_val = torch.utils.data.TensorDataset(X_train_valid, y_train_valid)
    dataloader_train = DataLoader(dst_train, batch_size=Batch_size, shuffle=True, num_workers=0, sampler=train_sampler)
    strong_unlabeldataloader_train = DataLoader(dst_strong_unlabeltrain, batch_size=Batch_size, shuffle=True,
                                                num_workers=0, sampler=train_sampler)
    weak_unlabeldataloader_train = DataLoader(dst_weak_unlabeltrain, batch_size=Batch_size, shuffle=True,
                                              num_workers=0, sampler=train_sampler)
    dataloader_train = DataLoader(dst_train, batch_size=Batch_size, shuffle=True, num_workers=0)
    strong_unlabeldataloader_train = DataLoader(dst_strong_unlabeltrain, batch_size=Batch_size, shuffle=True,
                                                num_workers=0)
    weak_unlabeldataloader_train = DataLoader(dst_weak_unlabeltrain, batch_size=Batch_size, shuffle=True,
                                              num_workers=0)
    dataloader_val = DataLoader(dst_val, shuffle=False, batch_size=Batch_size, num_workers=0)
    sum = 0
    sum1 = 0
    train_loss_sum = 0
    unsupervised_train_loss_sum = 0
    train_top1_sum = 0
    if Loss == 'CB_loss':
        torch.device('cuda')
        criterion = CB_loss(samples_per_cls=[5, 250, 50, 200, 50], no_of_classes=5, loss_type='focal', beta=0.9,
                            gamma=2.0, device=torch.device('cuda'))
    optimizer = t.optim.SGD(model.parameters(), momentum=0.9, weight_decay=5e-4, lr=Init_lr)
    sehcduler = select_sehcduler(Lr_scheduler, optimizer, Step_size, Multiplier, Total_epoch, Num_epochs)
    writer = SummaryWriter(OutpuDir)
    train_loss_list, val_loss_list = [], []
    train_top1_list, val_top1_list = [], []
    lr_list = []
    if IF_TRAIN:
        print('Start training')
        for epoch in range(Num_epochs):
            ep_start = time.time()
            model.train()
            top1_sum = 0
            top1_sum1 = 0
            train_opt.nowEpoch = epoch
            train_loader = zip(dataloader_train, strong_unlabeldataloader_train, weak_unlabeldataloader_train)
            for batch_idx, (data_l, data_s, data_w) in enumerate(train_loader):
                (ims, labels) = data_l
                ims = ims.permute(0, 3, 1, 2)
                if IF_GPU:
                    target = labels.cuda().long()
                    target = target - 1
                else:
                    target = labels.cpu().long()
                    target = target - 1
                (ims_weak, labels_weak) = data_w
                (ims_strong, labels_strong) = data_s
                ims_weak = ims_weak.permute(0, 3, 1, 2)
                ims_strong = ims_strong.permute(0, 3, 1, 2)
                if IF_GPU:
                    inputs = torch.cat((ims, ims_weak, ims_strong)).cuda()
                else:
                    inputs = torch.cat((ims, ims_weak, ims_strong)).cpu()
                label_batch_size = ims.shape[0]
                outputs = model(inputs)
                output = outputs[:label_batch_size]
                output_u_w, output_u_s = outputs[label_batch_size:].chunk(2)
                loss = criterion(output, target)
                pseudo_label = torch.softmax(output_u_w.detach_(), dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(PThreshold).float()
                max_probs = max_probs.max()
                Loss_u = (F.cross_entropy(output_u_s, targets_u, reduction='none') * mask).mean()
                loss = loss + Un_lamda * Loss_u
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                top1 = accuracy(output.data, target.data, topk=(1,))
                train_loss_sum += loss.data.cpu().numpy()
                unsupervised_train_loss_sum += Loss_u.data.cpu().numpy()
                train_top1_sum += top1[0].cpu().numpy()
                sum += 1
                top1_sum += top1[0].cpu().numpy()
            train_loss_list.append(train_loss_sum / sum)
            train_top1_list.append(train_top1_sum / sum)
            if (epoch + 1) % 10 == 0:
                lr = optimizer.state_dict()['param_groups'][0]['lr']  # 取得当前的lr
                print(
                    'Epoch [%d / %d ] | lr=[%f] | training loss [%.4f] | unsupervised loss [%.4f] | generated label prob [%.4f]' \
                    % (epoch + 1, Num_epochs, lr, train_loss_sum / sum, unsupervised_train_loss_sum / sum, max_probs)
                preds = np.asarray(preds)
                preds = preds + 1
            writer.close()
            lr_list.append(lr)
            sehcduler.step()
            sum = 0
            train_loss_sum = 0
            unsupervised_train_loss_sum = 0
            train_top1_sum = 0
            sum1 = 0
    else:
        top1_sum1 = 0
        sum1 = 0
        val_loss_sum = 0
        val_top1_sum = 0
        model.eval()
        with torch.no_grad():
            for im, (ims, labels) in enumerate(dataloader_val):
                if IF_GPU:
                    input = ims.cuda()
                    input = input.permute(0, 3, 1, 2)
                    target = labels.cuda().long()
                    target = target - 1
                else:
                    input = ims.cpu()
                    input = input.permute(0, 3, 1, 2)
                    target = labels.cpu().long()
                    target = target - 1

                output = model(input)
if __name__ == '__main__':
    if not os.path.isdir(OutpuDir):
        mkdir_p(OutpuDir)
    train()
