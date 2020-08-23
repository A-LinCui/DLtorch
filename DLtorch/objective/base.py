import abc

import torch

class BaseObjective(object):
    NAME = "BaseObjective"
    def __init__(self, criterion_type):
        self._criterion = getattr(torch.nn, criterion_type)()

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