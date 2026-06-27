import torchvision


def torchvision_transform(name, *args, **kwargs):
    return getattr(torchvision.transforms, name)(*args, **kwargs)
