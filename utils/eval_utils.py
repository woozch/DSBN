import torch
import numpy as np


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    num_samples = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.div_(num_samples))
    return res


def accuracy_of_c(output, target, class_idx, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    # num_samples = target.size(0)
    selection = target == class_idx
    target_selected = target[selection]
    output_selected = output[selection]
    num_samples = torch.sum(selection).float()

    _, pred = output_selected.topk(maxk, 1, True, True)
    pred = pred.t().float()
    correct = pred.eq((target_selected.view(1, -1).expand_as(pred)).float())

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.div_(num_samples))
    return res
