import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.train_utils import init_weights


class FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    # self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
    # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        # x = self.up_sample(x)
        # x = self.sigmoid(x)

        return x


class FCDiscriminator2(nn.Module):
    def __init__(self, num_classes, feat_dim=500):
        super(FCDiscriminator2, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, feat_dim, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(feat_dim, feat_dim, kernel_size=1, stride=1, padding=0)
        self.classifier = nn.Conv2d(feat_dim, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(F.relu(x))
        x = self.conv2(F.relu(x))
        x = self.classifier(x)

        return x


class DigitDiscriminator(nn.Module):
    def __init__(self, in_features):
        super(DigitDiscriminator, self).__init__()
        self.in_features = in_features

        self.discriminator = nn.Sequential(
            nn.Linear(self.in_features, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 1)
        )

        init_weights(self)

    def forward(self, x):
        return self.discriminator(x)


class OfficeDiscriminator(nn.Module):
    def __init__(self, in_features):
        super(OfficeDiscriminator, self).__init__()
        self.in_features = in_features

        self.discriminator = nn.Sequential(
            nn.Linear(self.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 1)
        )

        init_weights(self)
        last_layer = self.discriminator[6]
        last_layer.weight.data.normal_(0, 0.3).clamp_(min=-0.6, max=0.6)
        last_layer.bias.data.zero_()

    def forward(self, x):
        return self.discriminator(x)
    
    
class CPUADiscriminator(nn.Module):
    def __init__(self, in_features):
        super(CPUADiscriminator, self).__init__()
        self.in_features = in_features

        self.discriminator = nn.Sequential(
            nn.Linear(self.in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

        init_weights(self)
        last_layer = self.discriminator[4]
        last_layer.weight.data.normal_(0, 0.3).clamp_(min=-0.6, max=0.6)
        last_layer.bias.data.zero_()

    def forward(self, x):
        return self.discriminator(x)


class ProjectionDiscriminator(nn.Module):
    def __init__(self, num_classes, in_features):
        super(ProjectionDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.in_features = in_features

        self.fc1 = nn.Linear(in_features, in_features)
        self.fc2 = nn.Linear(in_features, in_features)
        self.embed = nn.Embedding(num_classes, in_features)
        self.linear = nn.Linear(in_features, 1)
        # self.fc1 = spectral_norm(nn.Linear(in_features, in_features))
        # self.fc2 = spectral_norm(nn.Linear(in_features, in_features))
        # self.embed = spectral_norm(nn.Embedding(num_classes, in_features))
        # self.linear = spectral_norm(nn.Linear(in_features, 1))

        init_weights(self)
        last_layer = self.linear
        last_layer.weight.data.normal_(0, 0.3).clamp_(min=-0.6, max=0.6)
        last_layer.bias.data.zero_()

    def forward(self, x, y):
        # x : (B, F), y: (B)
        x = self.fc1(x)
        x = self.fc2(F.relu(x))
        h = self.linear(x)  # (B, 1)
        embed = self.embed(y)  # (B, F)
        inner_prod = torch.sum(x * embed, dim=1, keepdim=True)  # (B, 1)
        return h + inner_prod
