# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import torch

from DLtorch.adv_attack import BaseAdvGenerator

class PGD(BaseAdvGenerator):
    def __init__(self, mean, std, epsilon, n_step, step_size, rand_init, criterion_type="CrossEntropyLoss",
                 criterion_kwargs=None):
        super(PGD, self).__init__(criterion_type, criterion_kwargs)
        self.mean = mean
        self.std = std
        self.rand_init = rand_init
        self.epsilon = epsilon
        self.n_step = n_step
        self.step_size = step_size

    def normalization_reverse(self, inputs):
        # By default, DLtorch framework will do normalization while loading datasets and the default pixel range isn't [0, 1].
        # Therefore, we should reverse the normalization process to adjust pixel value into [0, 1] before applying adversarial attack.
        # Mean and std used for initialize this adversary should be the same as that used for initializing dataset.
        input_channel = inputs.shape[1]
        mean = torch.reshape(torch.tensor(self.mean), (input_channel, 1, 1)).to(inputs.device)
        std = torch.reshape(torch.tensor(self.std), (input_channel, 1, 1)).to(inputs.device)
        inputs.data = inputs * std + mean
        return inputs

    def normalization(self, inputs):
        input_channel = inputs.shape[1]
        mean = torch.reshape(torch.tensor(self.mean), (input_channel, 1, 1)).to(inputs.device)
        std = torch.reshape(torch.tensor(self.std), (input_channel, 1, 1)).to(inputs.device)
        inputs.data = (inputs - mean) / std
        return inputs

    def generate_adv(self, net, inputs, targets, outputs=None):
        inputs_pgd = inputs.data.clone()
        inputs_pgd.requires_grad = True

        if self.rand_init:
            eta = inputs.new(inputs.size()).uniform_(-self.epsilon, self.epsilon)
            inputs_pgd = self.normalization_reverse(inputs_pgd)
            inputs_pgd.data = inputs_pgd + eta
            inputs_pgd = self.normalization(inputs_pgd)

        inputs = self.normalization_reverse(inputs)
        std = torch.reshape(torch.tensor(self.std), (inputs.shape[1], 1, 1)).to(inputs.device)
        step_size = self.step_size / std

        for _ in range(self.n_step):
            out = net(inputs_pgd)
            loss = self.criterion(out, targets)
            loss.backward()
            eta = step_size * inputs_pgd.grad.data.sign()
            inputs_pgd = inputs_pgd.data + eta
            inputs_pgd.requires_grad = True
            inputs_pgd = self.normalization_reverse(inputs_pgd)
            eta = torch.clamp(inputs_pgd.data - inputs, -self.epsilon, self.epsilon)
            inputs_pgd.data = inputs + eta
            inputs_pgd = self.normalization(inputs_pgd)

        inputs = self.normalization(inputs)
        net.zero_grad()
        return inputs_pgd.data