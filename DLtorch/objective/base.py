# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import abc

from DLtorch.component.criterion import get_criterion

class BaseObjective(object):
    NAME = "BaseObjective"
    def __init__(self, criterion_type, criterion_kwargs=None):
        self._criterion = get_criterion(criterion_type, **criterion_kwargs) \
            if criterion_kwargs is not None else get_criterion(criterion_type)

    # ---- virtual APIs to be implemented in subclasses ----
    @abc.abstractmethod
    def perf_names(self):
        """
        The names of the perf.
        """

    @abc.abstractmethod
    def get_perfs(self, inputs, outputs, targets, model):
        """
        Get the perfs.
        """

    @abc.abstractmethod
    def get_loss(self, inputs, outputs, targets, model):
        """
        Get the loss of a batch.
        """