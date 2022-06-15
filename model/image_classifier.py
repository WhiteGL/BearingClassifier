import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import utils
from utils.data_loader import get_data_loader


class ClassifierImage(nn.Module):
    """基于图片的故障诊断分类器，暂且只设计二分类"""
    def __init__(self, input_dim=1, output_dim=1, input_size=40):
        super(ClassifierImage, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x


class ImageClassifier(object):
    def __init__(self, args):
        """载入参数"""
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.gpu_mode = args.gpu_mode
        self.model_name = args.model_name
        self.input_size = args.input_size
        self.transforms = args.transforms
        self.normalize = args.normalize
        self.save_dir = args.save_dir


        # 载入数据集
        self.train_loader = get_data_loader(self.batch_size, self.transforms, self.normalize)
        self.val_loader = get_data_loader(self.batch_size, self.transforms, self.normalize, train=False)
        data = self.train_loader.__iter__().__next__()[0]
        # 初始化模型
        self.net = ClassifierImage(input_dim=1, output_dim=1, input_size=self.input_size)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        if self.gpu_mode:
            self.net.cuda()

        print('---------- Networks architecture -------------')
        utils.print_network(self.net)
        print('-----------------------------------------------')

    def train(self):
        self.train_hist = {}
        self.train_hist['train_loss'] = []
        self.train_hist['valid_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        training_loss = 0.0
        valid_loss = 0.0

        self.net.train()
        print('training start')
        start_time = time.time()
        for epoch in range(self.epoch):
            epoch_start_time = time.time()
            for iter, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                x, label = batch
                x = x.unsqueeze(1)
                x = x.cuda()
                label = label.cuda()
                y_ = self.net(x)
                loss = self.loss_fn(y_, label)
                loss.backward()
                self.optimizer.step()
                training_loss += loss.data.item() * x.size(0)
            training_loss /= len(self.train_loader.dataset)

            self.net.eval()
            num_correct = 0
            num_examples = 0
            for batch in self.val_loader:
                x, label = batch
                x = x.cuda()
                y_ = self.net(x)
                label = label.cuda()
                loss = self.loss_fn(y_, label)
                valid_loss += loss.data.item() * x.size(0)
                correct = torch.eq(torch.max(F.softmax(y_), dim=1)[1], label).view(-1)
                num_examples += correct.shape[0]
                num_correct += torch.sum(correct).item()
            valid_loss /= len(self.val_loader.dataset)

            self.train_hist['train_loss'].append(training_loss)
            self.train_hist['valid_loss'].append(valid_loss)

            print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'
                  .format(epoch, valid_loss, num_correct / num_examples))
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.model_name), self.model_name)

    def save(self):
        save_dir = os.path.join(self.save_dir, self.model_name)

        torch.save(self.net.state_dict(), os.path.join(save_dir, self.model_name + '.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.model_name)

        self.net.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '.pkl')))

        print('load model successful')