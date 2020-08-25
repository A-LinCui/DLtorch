# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import torch.optim

Optimizer = {}

def get_optimizer(_type, **kwargs):
    # Get an optimizer from DLtorch framework.
    if _type in Optimizer.keys():
        return Optimizer[_type](**kwargs)
    else:
        try:
            return getattr(torch.optim, _type)(**kwargs)
        except:
            print("No optimizer type: {}".format(_type))
            raise NameError

def regist_optimizer(name, fun):
    # Regist an optimizer into DLtorch framework.
    Optimizer[name] = fun

def get_optimizer_attrs():
    # Get all the optimizer types.
    # Used in "main.components".
    return list(Optimizer.keys())