import torch.optim as optim

def get_optimizer_cls(_type):
    return getattr(optim, _type)

def get_optimizer(_type, **kwargs):
    return get_optimizer_cls(_type)(**kwargs)