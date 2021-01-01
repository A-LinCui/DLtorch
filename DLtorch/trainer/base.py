# -*- coding:utf-8 -*-

import os
import abc

import torch

import DLtorch
from DLtorch.base import BaseComponent


class BaseTrainer(BaseComponent):
    def __init__(
        self,
        # Components
        model,
        dataset,
        dataloader_kwargs: dict,
        objective,
        optimizer_type: str,
        optimizer_kwargs: dict = {},
        lr_scheduler_type: str = None,
        lr_scheduler_kwargs: dict = {},
        # Training cfgs
        device: str = "cuda",
        epochs: int = 100,
        save_every: int = 10,
        save_as_state_dict: bool = True,
        report_every: int = 50,
        test_every: int = 1,
        grad_clip: float = 5.0,
        eval_no_grad: bool = True,
        ):
        super(BaseTrainer, self).__init__()
        self.logger.info("Trainer Constructed.")
        
        self.model = model
        self.dataset = dataset
        self.dataloader_kwargs = dataloader_kwargs
        self.objective = objective
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.device = device
        self.epochs = epochs
        self.save_every = save_every
        self.save_as_state_dict = save_as_state_dict
        self.report_every = report_every
        self.test_every = test_every
        self.grad_clip = grad_clip
        self.eval_no_grad = eval_no_grad

        # Init components
        self.optimizer = getattr(DLtorch.optimizer, self.optimizer_type)(**self.optimizer_kwargs, params=list(self.model.parameters()))
        self.lr_scheduler = getattr(DLtorch.lr_scheduler, self.lr_scheduler_type)(**self.lr_scheduler_kwargs, optimizer=self.optimizer) \
            if self.lr_scheduler_type is not None else None


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