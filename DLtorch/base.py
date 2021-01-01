# coding:utf-8

import sys
import os
import inspect
import ast

import torch.nn as nn

import DLtorch
from DLtorch.utils import *


class Plugins(object):
    def __init__(self, plugin_root: str):
        self.plugin_root = plugin_root
        
        for root, dirs, files in os.walk(self.plugin_root, True):
            module_list = self._load_modules(root)
            for module in module_list:
                if issubclass(module, DLtorch.objective.BaseObjective):
                    setattr(DLtorch.objective, module.__name__, module)
                elif issubclass(module, DLtorch.objective.adversary.BaseAdvGenerator):
                    setattr(DLtorch.objective.adversary, module.__name__, module)
                elif issubclass(module, DLtorch.dataset.BaseDataset):
                    setattr(DLtorch.dataset, module.__name__, module)
                elif issubclass(module, DLtorch.trainer.BaseTrainer):
                    setattr(DLtorch.trainer, module.__name__, module)
                elif issubclass(module, DLtorch.optimizer.BaseOptimizer):
                    setattr(DLtorch.optimizer, module.__name__, module)
                elif issubclass(module, DLtorch.criterion.BaseCriterion):
                    setattr(DLtorch.criterion, module.__name__, module)
                elif issubclass(module, DLtorch.lr_scheduler.BaseLrScheduler):
                    setattr(DLtorch.lr_scheduler, module.__name__, module)
                elif issubclass(module, DLtorch.model.BaseModel):
                    setattr(DLtorch.model, module.__name__, module)
        

    def _load_modules(self, root: str):
        """ 
        Dynamically load modules from all the files ending with '.py' under current root and return a list. 
        """

        modules = []
        for filename in os.listdir(root):
            if filename.endswith(".py"):
                name = os.path.splitext(filename)[0]
                filename = os.path.join(root, filename)
                if name.isidentifier():
                    fh = None
                    try:
                        fh = open(filename, "r", encoding="utf8")
                        code = fh.read()
                        module = type(sys)(name)
                        sys.modules[name] = module
                        exec(code, module.__dict__)
                        for _class in module.__dict__.values():
                            try:
                                if issubclass(_class, DLtorch.base.BaseComponent):
                                    modules.append(_class)
                            except:
                                continue
                    except (EnvironmentError, SyntaxError) as err:
                        sys.modules.pop(name, None)
                        print(err)            
                    finally:
                        if fh is not None:
                            fh.close()
        return modules


class BaseComponent(object):
    def __init__(self, **kwargs):
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
