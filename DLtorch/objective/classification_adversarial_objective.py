# -*- coding:utf-8 -*-

import collections

import torch

from DLtorch.utils import accuracy
from DLtorch.objective.base import BaseObjective
import DLtorch.objective.adversary as adversary


class ClassificationAdversarialObjective(BaseObjective):
    def __init__(
        self, 
        adversary_type: str = "PGD",
        adversary_kwargs: dict = {
            epsilon: 8 / 255, 
            n_step: 7, 
            step_size: 2 / 255, 
            rand_init: True, 
            criterion_type: "CrossEntropyLoss",
            criterion_kwargs: {},
            eval_mode: True
        },
        adv_loss_coef: float = 1.5, 
        adv_reward_coef: float = 0.5, 
        criterion_type: str = "CrossEntropyLoss", 
        criterion_kwargs: dict = {}
        ):
        super(ClassificationAdversarialObjective, self).__init__(criterion_type, criterion_kwargs)
        self.adversary_type = adversary_type
        self.adversary_kwargs = adversary_kwargs
        self.adversary = getattr(adversary, self.adversary_type)(**adversary_kwargs)
        self.adv_loss_coef = adv_loss_coef
        self.adv_reward_coef = adv_reward_coef
        self.adv_buffer = collections.OrderedDict()


    @ property
    def perf_names(self):
        return ["natrual_acc", "robust_acc"]


    def get_perfs(self, inputs, outputs, targets, model, **kwargs):
        if self.adv_loss_coef == 0:
            return [accuracy(outputs, targets)[0]]  # Top-1 accuracy
        else:
            adv_examples = self.generate_adv(model, inputs, targets, outputs)
            perfs = [accuracy(outputs, targets)[0], accuracy(model(adv_examples), targets)[0]]
            return perfs

    def generate_adv(self, model, inputs, targets, outputs=None):
        if inputs not in self.adv_buffer.keys():
            self.adv_buffer = collections.OrderedDict()
            self.adv_buffer[inputs] = self.adversary.generate_adv(model, inputs, targets, outputs)
        return self.adv_buffer[inputs]

    def get_loss(self, inputs, outputs, targets, model):
        if self.adv_loss_coef == 0:
            return self._criterion(outputs, targets)
        else:
            adv_examples = self.generate_adv(model, inputs, targets, outputs)
            natural_loss = self._criterion(outputs, targets)
            adv_loss = self._criterion(model(adv_examples), targets)
            return (1 - self.adv_loss_coef) * natural_loss + self.adv_loss_coef * adv_loss


    def get_reward(self, perf):
        return (1 - self.adv_reward_coef) * perf[0] + self.adv_reward_coef * perf[1]