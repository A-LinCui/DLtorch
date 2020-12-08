# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import abc

from torch.optim.optimizer import Optimizer, required

from DLtorch.base import BaseComponent

class BaseOptimizer(Optimizer, BaseComponent):
    def __init__(self, params, lr=required):
        defaults = dict(lr=lr)
        Optimizer.__init__(self, params, defaults)
        BaseComponent.__init__(self)