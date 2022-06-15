import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import utils
from utils.data_loader import get_data_loader
import matplotlib.pyplot as plt
import copy


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

        xb = xb.unsqueeze(1)

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
def show_loss_acc(num_epochs, loss_hist, metric_hist):
    # 损失值
    plt.title("Train-Val Loss")
    plt.plot(range(1, num_epochs+1), loss_hist["train"], label="train")
    plt.plot(range(1, num_epochs+1), loss_hist["val"], label="val")
    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")
    plt.legend()
    #plt.show()
    plt.savefig('./models/loss.png')
    plt.close()

    # 准确率
    plt.title("Train-Val Accuracy")
    plt.plot(range(1, num_epochs+1), metric_hist["train"], label="train")
    plt.plot(range(1, num_epochs+1), metric_hist["val"], label="val")
    plt.ylabel("Accuracy")
    plt.xlabel("Training Epochs")
    plt.legend()
    #plt.show()
    plt.savefig('./models/accuracy.png')
    plt.close()


class ClassifierImage(nn.Module):
    """基于图片的故障诊断分类器，暂且只设计二分类"""
    def __init__(self, params):
        super(ClassifierImage, self).__init__()
        c_in, h_in, w_in = params["input_shape"]
        init_f = params["initial_filters"]
        num_fc1 = params["num_fc1"]
        num_classes = params["num_classes"]
        self.dropout_rate = params["dropout_rate"]

        self.conv1 = nn.Conv2d(c_in, init_f, kernel_size=3)
        h, w = get_conv2d_out_shape(h_in, w_in, self.conv1)

        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3)
        h, w = get_conv2d_out_shape(h, w, self.conv2)

        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3)
        h, w = get_conv2d_out_shape(h, w, self.conv3)

        self.num_flatten = h * w * 4 * init_f

        self.fc1 = nn.Linear(self.num_flatten, num_fc1)

        self.fc2 = nn.Linear(num_fc1, num_classes)

        utils.initialize_weights(self)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, self.num_flatten)
        x = F.relu(self.fc1(x))

        x = F.dropout(x, self.dropout_rate)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class ImageClassifier(object):
    def __init__(self, args):
        """载入参数"""
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.gpu_mode = args.gpu_mode
        self.transforms = args.transforms
        self.normalize = args.normalize

        if self.gpu_mode:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 模型相关参数值
        params_model = {
            "input_shape": (1, 40, 40),  # C*H*W
            "initial_filters": 8,
            "num_fc1": 100,
            "dropout_rate": 0.25,
            "num_classes": 2,
        }

        # 载入数据集
        self.train_loader = get_data_loader(self.batch_size, self.transforms, self.normalize)
        self.val_loader = get_data_loader(self.batch_size, self.transforms, self.normalize, train=False)
        # 初始化模型
        self.net = ClassifierImage(params_model)
        self.opt = optim.Adam(self.net.parameters(), lr=0.001)
        self.loss_fn = torch.nn.NLLLoss(reduction='sum')
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min',factor=0.5, patience=20,
                                                                 verbose=True)
        self.params_train = {
            "num_epochs": self.epoch,
            "optimizer": self.opt,
            "loss_func": self.loss_fn,
            "train_dl": self.train_loader,
            "val_dl": self.val_loader,
            "sanity_check": True,
            "lr_scheduler": self.lr_scheduler,
            "path2weights": "./models/bic/weights.pt",
        }

        if self.gpu_mode:
            self.net.to(self.device)

        print('---------- Networks architecture -------------')
        utils.print_network(self.net)
        print('-----------------------------------------------')

    def train_val(self):

        model = self.net
        params = self.params_train
        # 取得各参数值
        num_epochs = params["num_epochs"]
        loss_func = params["loss_func"]
        opt = params["optimizer"]
        train_dl = params["train_dl"]
        val_dl = params["val_dl"]
        sanity_check = params["sanity_check"]
        lr_scheduler = params["lr_scheduler"]
        path2weights = params["path2weights"]

        # 存储每轮次损失值
        loss_history = {
            "train": [],
            "val": [],
        }

        # 存储每轮次正确数
        metric_history = {
            "train": [],
            "val": [],
        }

        # 深度复制的最优模型参数
        best_model_wts = copy.deepcopy(model.state_dict())

        # 初始化损失值
        best_loss = float('inf')

        # 主循环代码块
        for epoch in range(num_epochs):

            # 取得当前学习率值
            current_lr = get_lr(opt)
            print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))

            # 定义为模型训练阶段
            model.train()
            train_loss, train_metric = loss_epoch(model, loss_func, train_dl, self.device,
                                                  sanity_check, opt)

            # 存储训练各轮次结果值
            loss_history["train"].append(train_loss)
            metric_history["train"].append(train_metric)

            # 模型验证阶段
            model.eval()
            with torch.no_grad():
                val_loss, val_metric = loss_epoch(model, loss_func, val_dl, self.device,
                                                  sanity_check)

            # 存储过程中最好的结果数据
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())

                # 将参数存储到本地
                torch.save(model.state_dict(), path2weights)
                print("Copied best model weights!")

            # 存储验证过程各轮次结果值
            loss_history["val"].append(val_loss)
            metric_history["val"].append(val_metric)

            # 学习率更新策略
            lr_scheduler.step(val_loss)
            if current_lr != get_lr(opt):
                print("Loading best model weights!")
                model.load_state_dict(best_model_wts)

            print("train loss: %.6f, dev loss: %.6f, accuracy: %.2f" % (train_loss, val_loss, 100 * val_metric))
            print("-" * 10)

            # 加载整个过程中最好的参数
        model.load_state_dict(best_model_wts)

        return model, loss_history, metric_history