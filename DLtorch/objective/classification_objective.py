import torch.nn as nn
from DLtorch.utils.torch_utils import accuracy
from DLtorch.objective.base import BaseObjective

class ClassificationObjective(BaseObjective):
    NAME = "ClassificationObjective"

    def __init__(self):
        super(ClassificationObjective, self).__init__()
        self._criterion = nn.CrossEntropyLoss()

    def perf_names(self):
        return ["acc"]

    def get_perfs(self, inputs, outputs, targets, model, **kwargs):
        return [float(accuracy(outputs, targets))]

    def get_loss(self, inputs, outputs, targets, model):
        return self._criterion(outputs, targets)