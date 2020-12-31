# -*- coding: utf-8 -*-

import abc

from DLtorch.base import BaseComponent
import DLtorch.criterion


class BaseObjective(BaseComponent):
    def __init__(
        self, 
        criterion_type: str = "CrossEntropyLoss", 
        criterion_kwargs: dict = {}
        ):
        super(BaseObjective, self).__init__()

        self._criterion = getattr(DLtorch.criterion, criterion_type)(**criterion_kwargs)

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
