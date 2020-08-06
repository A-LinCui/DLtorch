import torch.optim.lr_scheduler

Scheduler = {}

def get_scheduler(_type, **kwargs):
    if _type in Scheduler.keys():
        return Scheduler[_type](**kwargs)
    else:
        return getattr(torch.optim.lr_scheduler, _type)(**kwargs)

def regist_scheduler(name, fun):
    Scheduler[name] = fun