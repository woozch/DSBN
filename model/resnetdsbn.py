import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from model.dsbn import DomainSpecificBatchNorm2d
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _ntuple

from collections import OrderedDict
import operator
from itertools import islice

_pair = _ntuple(2)

__all__ = ['resnet18dsbn', 'resnet34dsbn', 'resnet50dsbn', 'resnet101dsbn', 'resnet152dsbn']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# class Conv2d(nn.Conv2d):
#     def __init__(self, *args, **kwargs):
#         super(Conv2d, self).__init__(*args, **kwargs)
#
#     def forward(self, x, domain_label):
#         return F.conv2d(x, self.weight, self.bias, self.stride,
#                          self.padding, self.dilation, self.groups), domain_label

class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode='zeros')

    def forward(self, input, domain_label):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups), domain_label


class TwoInputSequential(nn.Module):
    r"""A sequential container forward with two inputs.
    """

    def __init__(self, *args):
        super(TwoInputSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return TwoInputSequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(TwoInputSequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input1, input2):
        for module in self._modules.values():
            input1, input2 = module(input1, input2)
        return input1, input2

    
def resnet18dsbn(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DSBNResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        updated_state_dict = _update_initial_weights_dsbn(model_zoo.load_url(model_urls['resnet18']),
                                                          num_classes=model.num_classes,
                                                          num_domains=model.num_domains)
        model.load_state_dict(updated_state_dict, strict=False)

    return model


def resnet34dsbn(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DSBNResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        updated_state_dict = _update_initial_weights_dsbn(model_zoo.load_url(model_urls['resnet34']),
                                                          num_classes=model.num_classes,
                                                          num_domains=model.num_domains)
        model.load_state_dict(updated_state_dict, strict=False)

    return model

def resnet50dsbn(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DSBNResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        updated_state_dict = _update_initial_weights_dsbn(model_zoo.load_url(model_urls['resnet50']),
                                                          num_classes=model.num_classes,
                                                          num_domains=model.num_domains)
        model.load_state_dict(updated_state_dict, strict=False)

    return model


def resnet101dsbn(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DSBNResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        updated_state_dict = _update_initial_weights_dsbn(model_zoo.load_url(model_urls['resnet101']),
                                                          num_classes=model.num_classes,
                                                          num_domains=model.num_domains)
        model.load_state_dict(updated_state_dict, strict=False)

    return model


def resnet152dsbn(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DSBNResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        updated_state_dict = _update_initial_weights_dsbn(model_zoo.load_url(model_urls['resnet152']),
                                                          num_classes=model.num_classes,
                                                          num_domains=model.num_domains)
        model.load_state_dict(updated_state_dict, strict=False)

    return model


def _update_initial_weights_dsbn(state_dict, num_classes=1000, num_domains=2, dsbn_type='all'):
    new_state_dict = state_dict.copy()

    for key, val in state_dict.items():
        update_dict = False
        if ((('bn' in key or 'downsample.1' in key) and dsbn_type == 'all') or
                (('bn1' in key) and dsbn_type == 'partial-bn1')):
            update_dict = True

        if (update_dict):
            if 'weight' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-6] + 'bns.{}.weight'.format(d)] = val.data.clone()

            elif 'bias' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-4] + 'bns.{}.bias'.format(d)] = val.data.clone()

            if 'running_mean' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-12] + 'bns.{}.running_mean'.format(d)] = val.data.clone()

            if 'running_var' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-11] + 'bns.{}.running_var'.format(d)] = val.data.clone()

            if 'num_batches_tracked' in key:
                for d in range(num_domains):
                    new_state_dict[
                        key[0:-len('num_batches_tracked')] + 'bns.{}.num_batches_tracked'.format(d)] = val.data.clone()

    if num_classes != 1000 or len([key for key in new_state_dict.keys() if 'fc' in key]) > 1:
        key_list = list(new_state_dict.keys())
        for key in key_list:
            if 'fc' in key:
                print('pretrained {} are not used as initial params.'.format(key))
                del new_state_dict[key]

    return new_state_dict


class DSBNResNet(nn.Module):
    def __init__(self, block, layers, in_features=256, num_classes=1000, num_domains=2):
        self.inplanes = 64
        self.in_features = in_features
        self.num_domains = num_domains
        self.num_classes = num_classes
        super(DSBNResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = DomainSpecificBatchNorm2d(64, self.num_domains)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], num_domains=self.num_domains)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, num_domains=self.num_domains)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, num_domains=self.num_domains)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, num_domains=self.num_domains)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        if self.in_features != 0:
            self.fc1 = nn.Linear(512 * block.expansion, self.in_features)
            self.fc2 = nn.Linear(self.in_features, num_classes)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, num_domains=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = TwoInputSequential(
                Conv2d(self.inplanes, planes * block.expansion,
                       kernel_size=1, stride=stride, bias=False),
                DomainSpecificBatchNorm2d(planes * block.expansion, num_domains),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num_domains=num_domains))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, num_domains=num_domains))

        return TwoInputSequential(*layers)

    def forward(self, x, domain_label, with_ft=False):
        x = self.conv1(x)
        x, _ = self.bn1(x, domain_label)
        x = self.relu(x)
        x = self.maxpool(x)
        x, _ = self.layer1(x, domain_label)
        x, _ = self.layer2(x, domain_label)
        x, _ = self.layer3(x, domain_label)
        x, _ = self.layer4(x, domain_label)

        x = x.mean(3).mean(2)  # global average pooling
        x = x.view(x.size(0), -1)
        if self.in_features != 0:
            x = self.fc1(x)
            feat = x
            x = self.fc2(x)
        else:
            x = self.fc(x)
            feat = x
        if with_ft:
            return x, feat
        else:
            return x

        
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_domains=2):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = DomainSpecificBatchNorm2d(planes, num_domains)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = DomainSpecificBatchNorm2d(planes, num_domains)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, domain_label):
        residual = x

        out = self.conv1(x)
        out, _ = self.bn1(out, domain_label)
        out = self.relu(out)

        out = self.conv2(out)
        out, _ = self.bn2(out, domain_label)

        if self.downsample is not None:
            residual, _ = self.downsample(x, domain_label)

        out += residual
        out = self.relu(out)

        return out, domain_label


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_domains=2):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = DomainSpecificBatchNorm2d(planes, num_domains)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = DomainSpecificBatchNorm2d(planes, num_domains)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = DomainSpecificBatchNorm2d(planes * 4, num_domains)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, domain_label):
        residual = x

        out = self.conv1(x)
        out, _ = self.bn1(out, domain_label)
        out = self.relu(out)

        out = self.conv2(out)
        out, _ = self.bn2(out, domain_label)
        out = self.relu(out)

        out = self.conv3(out)
        out, _ = self.bn3(out, domain_label)

        if self.downsample is not None:
            residual, _ = self.downsample(x, domain_label)

        out += residual
        out = self.relu(out)

        return out, domain_label
