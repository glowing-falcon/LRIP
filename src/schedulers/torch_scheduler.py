import torch


def torch_scheduler(name, optimizer, *args, **kwargs):
    return getattr(torch.optim.lr_scheduler, name)(optimizer, *args, **kwargs)
