# coding:utf-8
# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import sys
import os
import inspect

import torch.nn as nn

import DLtorch
from DLtorch.utils import *

class Plugins(object):
    def __init__(self, plugin_root: str):
        self.plugin_root = plugin_root
        
        for root, dirs, files in os.walk(self.plugin_root, True):
            module_list = self._load_modules(root)
            for modules in module_list:
                modules = inspect.getmembers(modules, inspect.isclass)
                for module in modules:
                    if issubclass(module, DLtorch.objective.BaseObjective):
                        setattr(DLtorch.objective, module.__name__, module)
                    elif issubclass(module, DLtorch.objective.adversary.BaseAdvGenerator):
                        setattr(DLtorch.objective.adversary, module.__name__, module)
                    elif issubclass(module, DLtorch.datasets.BaseDataset):
                        setattr(DLtorch.datasets, module.__name__, module)
                    elif issubclass(module, DLtorch.trainer.BaseFinalTrainer):
                        setattr(DLtorch.trainer, module.__name__, module)
                    elif issubclass(module, DLtorch.optimizer.BaseOptimizer):
                        setattr(DLtorch.optimizer, module.__name__, module)
                    elif issubclass(module, DLtorch.criterion.BaseCriterion):
                        setattr(DLtorch.criterion, module.__name__, module)
                    elif issubclass(module, DLtorch.lr_scheduler.BaseLrScheduler):
                        setattr(DLtorch.lr_scheduler, module.__name__, module)
                    elif issubclass(module, nn.Module):
                        setattr(DLtorch.models, module.__name__, module)
        
    def _load_modules(self, root: str):
        """ Dynamically load modules from all the files ending with '.py' under current root and return a list. """

        modules = []
        for filename in os.listdir(root):
            if filename.endswith(".py"):
                name = os.path.splitext(filename)[0]
                if name.isidentifier():
                    fh = None
                    try:
                        fh = open(filename, "r", encoding="utf8")
                        code = fh.read()
                        module = type(sys)(name)
                        sys.modules[name] = module
                        exec(code, module.__dict__)
                        modules.append(module)
                    except (EnvironmentError, SyntaxError) as err:
                        sys.modules.pop(name, None)
                        print(err)
                    finally:
                        if fh is not None:
                            fh.close()
        return modules

class BaseComponent(object):
    def __init__(self):
        self._logger = None
    
    @property
    def logger(self):
        if self._logger is None:
            self._logger = getLogger(self.__class__.__name__)
        return self._logger

    def __getstate__(self):
        state = self.__dict__.copy()
        if "_logger" in state:
            del state["_logger"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # set self._logger to None
        self._logger = None