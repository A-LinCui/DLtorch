# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

from DLtorch.utils import accuracy
from DLtorch.objective.base import BaseObjective
from DLtorch.component import regist_objective

class ExampleNewObjective(BaseObjective):
    NAME = "ExampleNewObjective"

    def __init__(self, criterion_type="CrossEntropyLoss", criterion_kwargs=None):
        super(ExampleNewObjective, self).__init__(criterion_type, criterion_kwargs)

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


# Note that "register" function will be ran to regist all new components into DLtorch framework."
# We define value API to register different kinds of components.
"""
from DLtorch.component import regist_objective
from DLtorch.component import regist_scheduler
from DLtorch.component import regist_model
from DLtorch.component import regist_objective
from DLtorch.component import regist_optimizer
from DLtorch.component import regist_trainer
"""

def register():
    regist_objective("ExampleNewObjective", ExampleNewObjective)