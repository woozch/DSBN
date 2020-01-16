import torch
from torch import nn as nn


class Centroids(nn.Module):
    def __init__(self, feature_dim, num_classes, decay_const=0.3):
        super(Centroids, self).__init__()
        self.decay_const = decay_const
        self.num_classes = num_classes
        self.centroids = nn.Parameter(torch.randn(num_classes, feature_dim))
        self.centroids.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        self.centroids.data.zero_()

    def forward(self, x, y, y_mask=None):
        classes = torch.unique(y)
        current_centroids = []
        for c in range(self.num_classes):
            if c in classes:
                if y_mask is not None:
                    avg_c = torch.sum(x[(y == c) & y_mask, :], dim=0) / torch.sum((y == c) & y_mask).float()
                else:
                    avg_c = torch.sum(x[(y == c), :], dim=0) / torch.sum((y == c)).float()
                # avg_c = torch.mean(x[y == c, :], dim=0, keepdim=True)
                current_centroids.append(avg_c * self.decay_const + (1 - self.decay_const) * self.centroids[c:c + 1, :])
            else:
                # current_centroids.append(torch.zeros_like(self.centroids[c:c + 1, :]))
                current_centroids.append(self.centroids[c:c + 1, :])
        current_centroids = torch.cat(current_centroids, 0)
        return current_centroids
