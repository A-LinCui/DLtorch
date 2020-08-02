import torch.nn as nn

class BaseObjective(object):
    NAME = "BaseObjective"
    def __init__(self):
        self._criterion = nn.CrossEntropyLoss()

    def perf_names(self):
        pass

    def get_perfs(self, inputs, outputs, targets, model):
        pass

    def get_loss(self, inputs, outputs, targets, model):
        pass