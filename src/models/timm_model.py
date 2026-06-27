import logging

import timm
import torch


def timm_model(*args, **kwargs):
    model = timm.create_model(**kwargs)
    return model


def timm_model_autoclasses(dataset_info, *args, seed=None, **kwargs):
    kwargs["num_classes"] = dataset_info["num_classes"]
    if seed is None:
        model = timm.create_model(**kwargs)
    else:
        if kwargs.get("pretrained", False):
            logging.warning("Pretrained weights will be loaded. Why are you setting a seed?")
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            model = timm.create_model(**kwargs)
    return model
