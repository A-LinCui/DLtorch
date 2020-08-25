# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import torch

from DLtorch.adv_attack import BaseAdvGenerator

class FGSM(BaseAdvGenerator):
    def __init__(self, mean, std, epsilon, rand_init, criterion_type="CrossEntropyLoss", criterion_kwargs=None):
        super(FGSM, self).__init__(criterion_type, criterion_kwargs)
        self.mean = mean
        self.std = std
        self.epsilon = epsilon
        self.rand_init = rand_init

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
        if self.rand_init:
            eta = inputs.new(inputs.size()).uniform_(-self.epsilon, self.epsilon)
            inputs = self.normalization_reverse(inputs)
            inputs.data = inputs + eta
            inputs = self.normalization(inputs)

        std = torch.reshape(torch.tensor(self.std), (inputs.shape[1], 1, 1)).to(inputs.device)
        step_size = self.epsilon / std
        inputs.requires_grad = True
        outputs = net(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        eta = step_size * inputs.grad.data.sign()
        inputs.data = inputs + eta
        inputs = self.normalization_reverse(inputs)
        inputs = torch.clamp(inputs.data, 0., 1.0)
        inputs = self.normalization(inputs)
        net.zero_grad()
        return inputs.data