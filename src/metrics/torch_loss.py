import torch


def torch_loss(name, *args, **kwargs):
    return getattr(torch.nn, name)(*args, **kwargs)
