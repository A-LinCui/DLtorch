# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable

from DLtorch.objective.adversary import BaseAdvGenerator


class PGD(BaseAdvGenerator):
    """
    Project Gradient Descent
    Madry, Aleksander, et al. "Towards deep learning models resistant to adversarial attacks." arXiv preprint arXiv:1706.06083 (2017).
    """

    def __init__(
        self, 
        epsilon: float = 8 / 255, 
        n_step: int = 7, 
        step_size: float = 2 / 255, 
        rand_init: bool = True, 
        criterion_type: str = "CrossEntropyLoss",
        criterion_kwargs: dict = {}, 
        eval_mode: bool = True
        ):
        super(PGD, self).__init__(criterion_type, criterion_kwargs, eval_mode)
        
        self.epsilon = epsilon
        self.n_step = n_step
        self.step_size = step_size
        self.rand_init = rand_init

    def generate_adv(self, net, inputs, targets, outputs=None):
        if self.eval_mode:
            net_training_mode = net.training
            net.eval()
        else:
            net_training_mode = False

        inputs_pgd = inputs.data.clone()
        inputs_pgd.requires_grad = True

        if self.rand_init:
            eta = inputs.new(inputs.size()).uniform_(-self.epsilon, self.epsilon)
            inputs_pgd.data = inputs_pgd + eta

        for _ in range(self.n_step):
            out = net(inputs_pgd)
            loss = self._criterion(out, targets)
            loss.backward()
            eta = self.step_size * inputs_pgd.grad.data.sign()
            inputs_pgd = Variable(inputs_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(inputs_pgd.data - inputs, -self.epsilon, self.epsilon)
            inputs_pgd.data = inputs + eta
            inputs_pgd.data = torch.clamp(inputs_pgd.data, 0.0, 1.0)

        if net_training_mode:
            net.train()

        return inputs_pgd.data
