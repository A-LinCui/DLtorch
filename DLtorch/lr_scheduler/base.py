# -*- coding:utf-8 -*-

import abc

from torch.optim.lr_scheduler import _LRScheduler

from DLtorch.base import BaseComponent


class BaseLrScheduler(_LRScheduler, BaseComponent):
    def __init__(self, optimizer, last_epoch: int = -1):
        _LRScheduler.__init__(self, optimizer, last_epoch)
        BaseComponent.__init__(self)
    
    @abc.abstractmethod
    def get_lr(self):
        """ Get updated learning rate."""