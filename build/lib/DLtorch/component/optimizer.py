import torch.optim

Optimizer = {}

def get_optimizer(_type, **kwargs):
    if _type in Optimizer.keys():
        return Optimizer[_type](**kwargs)
    else:
        return getattr(torch.optim, _type)(**kwargs)

def regist_optimizer(name, fun):
    Optimizer[name] = fun