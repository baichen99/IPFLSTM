
import torch

def mean_absolute_error(pred, target):
    return torch.mean(torch.abs(pred - target))

def mean_squared_error(pred, target):
    return torch.mean((pred - target) ** 2)

def relative_error(pred, target):
    return torch.mean(torch.abs((pred - target) / (target + 1e-6)))
