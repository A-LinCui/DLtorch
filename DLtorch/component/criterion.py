# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import torch

Criterion = {}

def get_criterion(_type, **kwargs):
    # Get a criterion from DLtorch framework.
    if _type in Criterion.keys():
        return Criterion[_type](**kwargs)
    else:
        try:
            return getattr(torch.nn, _type)(**kwargs)
        except:
            print("No criterion type: {}".format(_type))
            raise NameError

def regist_Criterion(name, fun):
    # Regist a criterion into DLtorch Framework.
    Criterion[name] = fun

def get_criterion_attrs():
    # Get all the crierion types.
    # Used in "main.components".
    return list(Criterion.keys())