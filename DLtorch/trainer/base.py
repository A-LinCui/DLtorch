# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import os
import abc

import torch

from DLtorch.base import BaseComponent
from DLtorch.utils.torch_utils import get_params

class BaseTrainer(BaseComponent):
    def __init__(
        self,
        # Components
        model,
        dataset,
        dataloader_kwargs: dict,
        objective,
        optimizer,
        optimizer_kwargs: dict,
        lr_scheduler,
        lr_scheduler_kwargs: dict,
        save_every: int = 20,
        save_as_state_dict: bool = True,
        report_every: int = 50,
        grad_clip: float = None,
        eval_no_grad: bool = True
        ):
        super(BaseTrainer, self).__init__()
        
        self.model = model
        self.dataset = dataset
        self.dataloader_kwargs = dataloader_kwargs
        self.objective = objective
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.epochs = epochs
        self.save_every = save_every
        self.report_every = report_every
        self.grad_clip = grad_clip
        self.eval_no_grad = eval_no_grad

    # ---- virtual APIs to be implemented in subclasses ----
    @abc.abstractmethod
    def train(self):
        """
        Do the actual training task of your trainer.
        """

    @abc.abstractmethod
    def test(self, dataset):
        """
        Test the newest model on different datasets.
        """

    @abc.abstractmethod
    def save(self, path):
        """
        Save the trainer state to disk.
        """

    @abc.abstractmethod
    def load(self, path):
        """
        Load the trainer state from disk.
        """

    @abc.abstractmethod
    def infer(self, data_queue, _type):
        """
        Infer the model.
        """

    # ---- Construction Helper ---
    def count_param(self, only_trainable=False):
        """
        Count the parameter number for the model.
        """
        self.param = get_params(self.model, only_trainable=only_trainable)
        self.log.info("Parameter number for current model: {}M".format(self.param / 1.0e6))
