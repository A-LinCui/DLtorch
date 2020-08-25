# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import torch.optim.lr_scheduler

Scheduler = {}

def get_scheduler(_type, **kwargs):
    # Get a lr_scheduler from DLtorch framework.
    if _type in Scheduler.keys():
        return Scheduler[_type](**kwargs)
    else:
        try:
            return getattr(torch.optim.lr_scheduler, _type)(**kwargs)
        except:
            print("No lr_scheduler: {}".format(_type))
            raise NameError

def regist_scheduler(name, fun):
    # Regist an lr_scheduler into DLtorch Framework.
    Scheduler[name] = fun

def get_scheduler_attrs():
    # Get all the dataset types.
    # Used in "main.components".
    return list(Scheduler.keys())