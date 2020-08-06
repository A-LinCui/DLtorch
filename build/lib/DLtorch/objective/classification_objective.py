import torch.nn as nn
from DLtorch.utils.torch_utils import correct
from DLtorch.objective.base import BaseObjective

class ClassificationObjective(BaseObjective):
    NAME = "ClassificationObjective"

    def __init__(self):
        super(ClassificationObjective, self).__init__()
        self._criterion = nn.CrossEntropyLoss()

    @ property
    def perf_names(self):
        return ["correct"]

    def get_perfs(self, inputs, outputs, targets, model, **kwargs):
        return [correct(outputs, targets)]

    def get_loss(self, inputs, outputs, targets, model):
        return self._criterion(outputs, targets)

    def get_reward(self, perf):
        # By default, consider the first item in perf as reward
        return perf[0]