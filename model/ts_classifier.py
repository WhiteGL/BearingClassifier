import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import utils
from utils.data_loader import get_ts_data
from utils.utils import get_lr, loss_epoch
import copy


class ClassifierTS(nn.Module):
    """基于图片的故障诊断分类器，暂且只设计二分类"""
    def __init__(self, params):
        super(ClassifierTS, self).__init__()
        input_size = params["input_shape"]
        num_classes = params["num_classes"]
        self.dropout_rate = params["dropout_rate"]

        self.fc1 = nn.Linear(input_size, input_size // 2)

        self.fc2 = nn.Linear(input_size // 2, input_size // 4)

        self.fc3 = nn.Linear(input_size // 4, num_classes)

    def forward(self, input):
        x = F.relu(self.fc1(input))

        x = F.relu(self.fc2(x))

        x = F.dropout(x, self.dropout_rate)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)


class TSClassifier(object):
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
            "input_shape": 40,
            "dropout_rate": 0.25,
            "num_classes": 2,
        }

        # 载入数据集
        self.train_loader = get_ts_data(self.batch_size, self.normalize)
        self.val_loader = get_ts_data(self.batch_size, self.normalize, train=False)
        # 初始化模型
        self.net = ClassifierTS(params_model)
        self.opt = optim.Adam(self.net.parameters(), lr=0.001)
        self.loss_fn = torch.nn.NLLLoss(reduction='sum')
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', factor=0.5, patience=20,
                                                                 verbose=True)
        self.params_train = {
            "num_epochs": self.epoch,
            "optimizer": self.opt,
            "loss_func": self.loss_fn,
            "train_dl": self.train_loader,
            "val_dl": self.val_loader,
            "sanity_check": True,
            "lr_scheduler": self.lr_scheduler,
            "path2weights": "./models/tsnet/weights.pt",
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