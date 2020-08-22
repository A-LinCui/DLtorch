import torch
from torch.autograd import Variable

from DLtorch.adv_attack import BaseAdvGenerator

class FGSM(BaseAdvGenerator):
    def __init__(self, epsilon, rand_init, criterion_type="CrossEntropyLoss"):
        super(FGSM, self).__init__(criterion_type)
        self.epsilon = epsilon
        self.rand_init = rand_init

    def generate_adv(self, inputs, outputs, targets, net):
        if self.rand_init:
            eta = inputs.new(inputs.size()).uniform_(-self.epsilon, self.epsilon)
            inputs.data = inputs + eta

        outputs = net(inputs)
        loss = self.criterion(outputs, Variable(targets))
        loss.backward()
        eta = self.epsilon * inputs.grad.data.sign()
        inputs.data = inputs + eta
        inputs = torch.clamp(inputs.data, -1.0, 1.0)
        net.zero_grad()
        return inputs.data