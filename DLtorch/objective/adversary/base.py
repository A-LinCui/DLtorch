# -*- coding: utf-8 -*-

import abc
from collections import OrderedDict

import torch.nn as nn

import DLtorch
from DLtorch.base import BaseComponent


class BaseAdvGenerator(BaseComponent):
    def __init__(
        self, 
        criterion_type: str = "CrossEntropyLoss",
        criterion_kwargs: dict = {}, 
        eval_mode: bool = True
        ):
        super(BaseAdvGenerator, self).__init__()
        self.logger.info("Adversary Constructed.")
        self.criterion_type = criterion_type
        self.criterion_kwargs = criterion_kwargs
        self._criterion = getattr(DLtorch.criterion, self.criterion_type)(**self.criterion_kwargs)
        self.eval_mode = eval_mode

    @abc.abstractmethod
    def generate_adv(self, net, inputs, targets, outputs):
        """
        Generate adversarial types of the inputs.
        """
