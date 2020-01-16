import torch
from torch import nn as nn
from torch.nn import functional as F

from model.dsbn import DomainSpecificBatchNorm2d

from utils.train_utils import init_weights


class LeNet(nn.Module):
    """"Network used for MNIST or USPS experiments."""

    def __init__(self, num_classes=10, weights_init_path=None, in_features=0):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.in_features = in_features
        self.num_channels = 1
        self.image_size = 28
        self.name = 'LeNet'
        self.setup_net()

        if weights_init_path is not None:
            init_weights(self)
            self.load(weights_init_path)
        else:
            init_weights(self)

    def setup_net(self):
        self.conv1 = nn.Conv2d(self.num_channels, 20, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(20)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(50)
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        if self.in_features:
            self.fc2 = nn.Linear(500, self.in_features)
            self.fc3 = nn.Linear(self.in_features, self.num_classes)
        else:
            self.fc2 = nn.Linear(500, self.num_classes)

    def forward(self, x, with_ft=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(F.relu(x))
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(F.relu(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.in_features != 0:
            x = self.fc2(F.relu(x))
            feat = x
            x = self.fc3(x)
        else:
            x = self.fc2(F.relu(x))
            feat = x

        if with_ft:
            return x, feat
        else:
            return x

    def load(self, init_path):
        net_init_dict = torch.load(init_path)
        init_weights(self)
        updated_state_dict = self.state_dict()
        print('load {} params.'.format(init_path))
        for k, v in updated_state_dict.items():
            if k in net_init_dict:
                if v.shape == net_init_dict[k].shape:
                    updated_state_dict[k] = net_init_dict[k]
                else:
                    print(
                        "{0} params' shape not the same as pretrained params. Initialize with default settings.".format(
                            k))
            else:
                print("{0} params does not exist. Initialize with default settings.".format(k))
        self.load_state_dict(updated_state_dict)


class DSBNLeNet(nn.Module):
    """"Network used for MNIST or USPS experiments. Conditional Batch Normalization is added."""

    def __init__(self, num_classes=10, weights_init_path=None, in_features=0, num_domains=2):
        super(DSBNLeNet, self).__init__()
        self.num_classes = num_classes
        self.in_features = in_features
        self.num_channels = 1
        self.image_size = 28
        self.num_domains = num_domains
        self.name = 'DSBNLeNet'
        self.setup_net()

        if weights_init_path is not None:
            init_weights(self)
            self.load(weights_init_path)
        else:
            init_weights(self)

    def setup_net(self):
        self.conv1 = nn.Conv2d(self.num_channels, 20, kernel_size=5)
        self.bn1 = DomainSpecificBatchNorm2d(20, self.num_domains)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.bn2 = DomainSpecificBatchNorm2d(50, self.num_domains)
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        if self.in_features != 0:
            self.fc2 = nn.Linear(500, self.in_features)
            self.fc3 = nn.Linear(self.in_features, self.num_classes)
        else:
            self.fc2 = nn.Linear(500, self.num_classes)

    def forward(self, x, y, with_ft=False):
        x = self.conv1(x)
        x, _ = self.bn1(x, y)
        x = self.pool1(F.relu(x))
        x = self.conv2(x)
        x, _ = self.bn2(x, y)
        x = self.pool2(F.relu(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.in_features != 0:
            x = self.fc2(F.relu(x))
            feat = x
            x = self.fc3(x)
        else:
            x = self.fc2(F.relu(x))
            feat = x

        if with_ft:
            return x, feat
        else:
            return x

    def load(self, init_path):
        net_init_dict = torch.load(init_path)
        init_weights(self)
        updated_state_dict = self.state_dict()
        print('load {} params.'.format(init_path))
        for k, v in updated_state_dict.items():
            if k in net_init_dict:
                if v.shape == net_init_dict[k].shape:
                    updated_state_dict[k] = net_init_dict[k]
                else:
                    print(
                        "{0} params' shape not the same as pretrained params. Initialize with default settings.".format(
                            k))
            else:
                print("{0} params does not exist. Initialize with default settings.".format(k))
        self.load_state_dict(updated_state_dict)
