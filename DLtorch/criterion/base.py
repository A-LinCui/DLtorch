# -*- coding:utf-8 -*-

import abc

import torch.nn as nn

from DLtorch.base import BaseComponent


class BaseCriterion(nn.Module, BaseComponent):
    def __init__(self):
        nn.Module.__init__(self)
        BaseComponent.__init__(self)
        self.logger.info("Criterion Constructed.")
    
    @abc.abstractmethod
    def forward(self, inputs, targets):
        """ Get loss. """