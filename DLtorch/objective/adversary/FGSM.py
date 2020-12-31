# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from DLtorch.objective.adversary import BaseAdvGenerator


class FGSM(BaseAdvGenerator):
    """
    Fast Gradient Sign Method (FGSM)
    Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. "Explaining and harnessing adversarial examples." arXiv preprint arXiv:1412.6572 (2014).
    """

    def __init__(
        self,
        epsilon: float = 8 / 255, 
        criterion = nn.CrossEntropyLoss(), 
        eval_mode: bool = True
        ):
        super(FGSM, self).__init__(criterion, eval_mode)
        self.epsilon = epsilon

    def generate_adv(self, net: torch.nn.Module, inputs, targets, outputs=None):
        
        if self.eval_mode:
            net_training_mode = net.training
            net.eval()
        else:
            net_training_mode = False

        if outputs is None:
            outputs = net(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        eta = self.epsilon * inputs.grad.data.sign()
        inputs.data = inputs + eta
        inputs = torch.clamp(inputs.data, 0., 1.0)
        net.zero_grad()

        if net_training_mode:
            net.train()

        return inputs.data
