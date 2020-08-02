import torch.optim.lr_scheduler

def get_scheduler_cls(type_):
    return getattr(torch.optim.lr_scheduler, type_)

def get_scheduler(_type, **kwargs):
    return get_scheduler_cls(_type)(**kwargs)