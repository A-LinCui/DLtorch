import torch.nn as nn
from DLtorch.utils import accuracy
from DLtorch.objective.base import BaseObjective

class ClassificationObjective(BaseObjective):
    NAME = "ClassificationObjective"

    def __init__(self):
        super(ClassificationObjective, self).__init__()
        self._criterion = nn.CrossEntropyLoss()

    @ property
    def perf_names(self):
        return ["acc"]

    def get_perfs(self, inputs, outputs, targets, model, **kwargs):
        return [accuracy(outputs, targets)[0]]  # Top-1 accuracy

    def get_loss(self, inputs, outputs, targets, model):
        return self._criterion(outputs, targets)

    def get_reward(self, perf):
        # By default, consider the first item in perf as reward
        return perf[0]