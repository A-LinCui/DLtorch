# -*- coding: utf-8 -*-

import abc

import torch.nn as nn

from DLtorch.base import BaseComponent
import DLtorch.utils.torch_utils as torch_utils


class BaseModel(BaseComponent):
    def __init__(self, model_kwargs: dict = {}):
        super(BaseModel, self).__init__()
        self.model_kwargs = model_kwargs
        self.logger.info("Module Constructed.")
        self.logger.info("Parameters: {:.5f}M".format(torch_utils.get_params(self, only_trainable=False) / 1.e6))