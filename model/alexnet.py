import torch
from torch import nn as nn
from torch.nn import functional as F

from utils.train_utils import init_weights


class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1,
                                        padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, in_features=256, weights_init_path=None):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.in_features = in_features
        self.num_channels = 3
        self.image_size = 28
        self.name = 'AlexNet'
        self.dropout_keep_prob = 0.5

        self.setup_net()
        # if centroid:
        #     self.centroid = nn.Embedding(num_cls, self.out_dim)
        if weights_init_path is not None:
            init_weights(self)
            self.load(weights_init_path)
        else:
            init_weights(self)

        if self.in_features != 0:
            self.fc9.weight.data.normal_(0, 0.005).clamp_(min=-0.01, max=0.01)
            self.fc9.bias.data.fill_(0.1)
        else:
            self.fc8.weight.data.normal_(0, 0.005).clamp_(min=-0.01, max=0.01)
            self.fc8.bias.data.fill_(0.1)

    def setup_net(self):
        # 1st layer
        self.conv1 = nn.Conv2d(self.num_channels, 96, kernel_size=11, stride=4)
        self.pool1 = nn.MaxPool2d(3, stride=2)
        # self.norm1 = nn.LocalResponseNorm(96, alpha=1e-5, beta=0.75)
        self.norm1 = LRN(local_size=1, alpha=1e-5, beta=0.75)
        # 2nd layer
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, groups=2, padding=2)
        self.pool2 = nn.MaxPool2d(3, stride=2)
        # self.norm2 = nn.LocalResponseNorm(256, alpha=1e-5, beta=0.75)
        self.norm2 = LRN(local_size=1, alpha=1e-5, beta=0.75)
        # 3rd layer
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        # 4th layer
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, groups=2, padding=1)
        # 5th layer
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, groups=2, padding=1)
        self.pool5 = nn.MaxPool2d(3, stride=2)

        # classifier
        self.fc6 = nn.Linear(6 * 6 * 256, 4096)
        self.dropout1 = nn.Dropout(self.dropout_keep_prob)
        self.fc7 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(self.dropout_keep_prob)
        if self.in_features != 0:
            self.fc8 = nn.Linear(4096, self.in_features)
            self.fc9 = nn.Linear(self.in_features, self.num_classes)
        else:
            self.fc8 = nn.Linear(4096, self.num_classes)

    def forward(self, x, with_ft=False):
        # conv
        x = self.conv1(x)
        x = self.pool1(F.relu(x))
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.pool2(F.relu(x))
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.conv4(F.relu(x))
        x = self.conv5(F.relu(x))
        x = self.pool5(F.relu(x))
        x = x.view(x.size(0), -1)
        # classifier
        x = self.fc6(x)
        x = self.dropout1(F.relu(x))
        x = self.fc7(x)
        x = self.dropout2(F.relu(x))
        # feature layer without relu
        if self.in_features != 0:
            x = self.fc8(x)
            feature = x
            score = self.fc9(x)
        else:
            score = self.fc8(x)
            feature = score

        if with_ft:
            return score, feature
        else:
            return score

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