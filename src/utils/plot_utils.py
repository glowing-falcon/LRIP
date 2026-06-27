from functools import reduce
from typing import Tuple, Union

import torch
import torch.nn.utils.prune as prune
from torch.nn.modules.batchnorm import _BatchNorm


def prune_state_dict(state_dict):
    for name, module in state_dict.copy().items():
        if name.endswith("_orig"):
            # name: ...weight_orig -> ...weight
            name_prefix = name[:-5]
            mask = state_dict[name_prefix + "_mask"]
            state_dict[name_prefix] = module * mask
            del state_dict[name]
            del state_dict[name_prefix + "_mask"]
    return state_dict


def get_perturbation(
    traj: torch.Tensor,
    orig_vec: torch.Tensor,
    targ_coeffs: torch.Tensor,
    mask: torch.Tensor
):
    rhs = traj @ (traj.T @ targ_coeffs - orig_vec)
    lam = torch.linalg.solve((traj * mask) @ traj.T, rhs)
    return (traj.T @ lam) * mask


def filter_named_parameters(model, ignore_bn=False, ignore_bias=False):
    for name, param in model.named_parameters():
        if ignore_bn:
            module = reduce(getattr, name.split(".")[:-1], model)
            if isinstance(module, _BatchNorm):
                continue
        if ignore_bias:
            if name.endswith(".bias"):
                continue
        yield name, param


def flatten_model(
    model,
    consider_mask=False,
    ignore_bn=False,
    ignore_bias=False,
):
    weights = []
    for name, param in filter_named_parameters(model, ignore_bn, ignore_bias):
        if consider_mask and name.endswith("_orig"):
            attrs = name.split(".")
            attrs[-1] = attrs[-1][:-5] + "_mask"
            mask_param = reduce(getattr, attrs, model)
            weights.append((param * mask_param).flatten().clone())
        else:
            weights.append(param.flatten().clone())
    return torch.cat(weights)


def get_full_mask(
    model,
    ignore_bn=False,
    ignore_bias=False,
):
    weight_mask = []
    # NOTE: e.g. `weight_mask` will not show up, so rely on `weight_orig`
    for name, param in filter_named_parameters(model, ignore_bn, ignore_bias):
        if name.endswith("_orig"):
            attrs = name.split(".")
            attrs[-1] = attrs[-1][:-5] + "_mask"
            mask_param = reduce(getattr, attrs, model)
            weight_mask.append(mask_param.flatten())
        else:
            weight_mask.append(torch.ones_like(param).flatten())
    weight_mask = torch.cat(weight_mask)
    return weight_mask


def apply_correction(
    model,
    correction,
    ignore_bn=False,
    ignore_bias=False,
):
    weight_index = 0
    for name, param in filter_named_parameters(model, ignore_bn, ignore_bias):
        correction_layer = correction[weight_index:weight_index + param.numel()].view(param.shape)
        param.add_(correction_layer)
        weight_index += param.numel()
    assert weight_index == correction.numel()


def multi_dim_norm(
    x: torch.Tensor,
    dim: Union[int, Tuple[int, ...]],
    keepdim=False,
    restore_shape=False,
    eps=0.0
):
    """
    L2 norm over arbitrary dims with optional shape restoration.

    Args:
        x (Tensor): input tensor
        dim (int or tuple of ints): dims to reduce
        keepdim (bool): keep reduced dims (like PyTorch)
        restore_shape (bool): expand result back to original shape
        eps (float): numerical stability

    Returns:
        Tensor
    """
    if isinstance(dim, int):
        dim = (dim,)

    # Normalize dims
    dim = tuple(d % x.ndim for d in dim)

    # Compute norm
    norm = x.pow(2).sum(dim=dim, keepdim=keepdim).add(eps).sqrt()

    if restore_shape:
        if not keepdim:
            # reinsert singleton dims
            for d in sorted(dim):
                norm = norm.unsqueeze(d)
        # expand to original shape
        norm = norm.expand_as(x)

    return norm


def get_filter_norm(param: torch.Tensor):
    match param.dim():
        case 1:  # Bias, BN, etc
            norm = multi_dim_norm(param, dim=0, restore_shape=True)
        case 2:  # NN, etc
            norm = multi_dim_norm(param, dim=1, restore_shape=True)
        case 3:  # Tokens
            norm = multi_dim_norm(param, dim=(1, 2), restore_shape=True)
        case 4:  # CNN, etc
            norm = multi_dim_norm(param, dim=(1, 2, 3), restore_shape=True)
        case _:  # Unknown
            raise ValueError(f"Unsupported parameter dimension: {param.dim()}")
    return norm


def normalize_model(model, ignore_bn=False, ignore_bias=False):
    for name, param in filter_named_parameters(model, ignore_bn, ignore_bias):
        norm = get_filter_norm(param)
        param.div_(norm)


def mask_as_state_dict(state_dict, model):
    for name in state_dict.keys():
        if name.endswith("_mask"):
            # Create mask
            attrs = name.split(".")
            module = reduce(getattr, attrs[:-1], model)
            param_name = "_".join(attrs[-1].split("_")[:-1])
            prune.identity(module, param_name)
        elif name.endswith("_orig"):
            # Check that mask counterpart exists
            mask_name = "_".join(name.split("_")[:-1]) + "_mask"
            assert mask_name in state_dict
        else:
            # Remove mask if any
            attrs = name.split(".")
            module = reduce(getattr, attrs[:-1], model)
            param_name = "_".join(attrs[-1].split("_")[:-1])
            if hasattr(module, f"{param_name}_mask"):
                prune.remove(module, param_name)


def join_mask_and_weights(masks_state_dict, weights_state_dict):
    joined_state_dict = dict()
    for name in weights_state_dict:
        if f"{name}_mask" in masks_state_dict:
            joined_state_dict[f"{name}_mask"] = masks_state_dict[f"{name}_mask"]
            joined_state_dict[f"{name}_orig"] = weights_state_dict[name]
        else:
            joined_state_dict[name] = weights_state_dict[name]
    return joined_state_dict
