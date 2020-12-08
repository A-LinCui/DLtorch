# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import abc

import torch.optim

from DLtorch.base import BaseComponent

class BaseOptimizer(torch.optim, BaseComponent):
    def __init__(self):
        torch.optim.__init__(self)
        BaseComponent.__init__(self)