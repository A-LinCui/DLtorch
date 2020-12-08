# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

from torch.optim.lr_scheduler import _LRScheduler

from DLtorch.base import BaseComponent

class BaseLrScheduler(_LRScheduler, BaseComponent):
    def __init__(self):
        _LRScheduler.__init__(self)
        BaseComponent.__init__(self)