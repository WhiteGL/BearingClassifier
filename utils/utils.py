import os, gzip, torch
import torch.nn as nn
import numpy as np
import imageio
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def get_conv2d_out_shape(H_in, W_in, conv, pool=2):
    # get conv arguments
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation

    # Ref: https://pytorch.org/docs/stable/nn.html
    H_out = np.floor((H_in+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    W_out = np.floor((W_in+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

    if pool:
        H_out /= pool
        W_out /= pool
    return int(H_out),int(W_out)


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


def metrics_batch(output, target):
    # 取得预测输出类别
    pred = output.argmax(dim=1, keepdim=True)

    # 预测值与真实比较
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects


def loss_batch(loss_func, output, target, opt=None):
    # 取得损失值
    loss = loss_func(output, target)

    # 取得预测正确个数
    metric_b = metrics_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b


# 定义每轮次损失计算 epoch
def loss_epoch(model, loss_func, dataset_dl, device, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)

        #xb = xb.unsqueeze(1)
        yb = yb.long()

        output = model(xb)
        # 调用每批次损失计算
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

        # 更新损失值
        running_loss += loss_b

        # 叠加预测正确数
        if metric_b is not None:
            running_metric += metric_b

        # 在可用性检测条件下，跳出循环，即只循环一次batch
        if sanity_check is True:
            break

    # 计算损失平均值
    loss = running_loss / float(len_data)

    # 计算正确值平均
    metric = running_metric / float(len_data)

    return loss, metric


# 画出损失值与正确率
def show_loss_acc(num_epochs, loss_hist, metric_hist, model_name):
    # 损失值
    plt.title("Train-Val Loss")
    plt.plot(range(1, num_epochs+1), loss_hist["train"], label="train")
    plt.plot(range(1, num_epochs+1), loss_hist["val"], label="val")
    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")
    plt.legend()
    #plt.show()
    plt.savefig('./models/'+model_name+'/loss.png')
    plt.close()

    # 准确率
    plt.title("Train-Val Accuracy")
    plt.plot(range(1, num_epochs+1), metric_hist["train"], label="train")
    plt.plot(range(1, num_epochs+1), metric_hist["val"], label="val")
    plt.ylabel("Accuracy")
    plt.xlabel("Training Epochs")
    plt.legend()
    #plt.show()
    plt.savefig('./models/'+model_name+'/accuracy.png')
    plt.close()