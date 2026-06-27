import torch


def torch_optimizer(optimizer_name, params, *args, **kwargs):
    return getattr(torch.optim, optimizer_name)(params, *args, **kwargs)
