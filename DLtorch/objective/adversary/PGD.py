# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import torch
from torch.autograd import Variable

from DLtorch.objective.adversary import BaseAdvGenerator

class PGD(BaseAdvGenerator):
    def __init__(self, epsilon: float, n_step: int, step_size: float, rand_init: bool, criterion, eval_mode: bool):
        super(PGD, self).__init__(criterion, eval_mode)
        self.epsilon = epsilon
        self.n_step = n_step
        self.step_size = step_size
        self.rand_init = rand_init

    def generate_adv(self, net: torch.nn.Module, inputs, targets, outputs=None):
        
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
            loss = self.criterion(out, targets)
            loss.backward()
            eta = self.step_size * inputs_pgd.grad.data.sign()
            inputs_pgd = Variable(inputs_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(inputs_pgd.data - inputs, -self.epsilon, self.epsilon)
            inputs_pgd.data = inputs + eta
            inputs_pgd.data = torch.clamp(inputs_pgd.data, 0.0, 1.0)

        if net_training_mode:
            net.train()

        return inputs.data