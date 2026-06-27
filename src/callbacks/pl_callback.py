import pytorch_lightning as pl


def pl_callback(name, *args, **kwargs):
    return getattr(pl.callbacks, name)(*args, **kwargs)
