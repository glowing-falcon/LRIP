import torch


def get_attr_from_string(obj, attr_path):
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def get_modules(model, include, ignore=()):
    included_modules = [get_attr_from_string(torch, mod) for mod in include]
    target_moduledict = {}
    for name, module in model.named_modules():
        if isinstance(module, tuple(included_modules)) and (name not in ignore):
            target_moduledict[f"{name}.weight"] = (module, "weight")
    return target_moduledict
