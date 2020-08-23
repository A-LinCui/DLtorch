import torch
from torch.autograd import Variable

from DLtorch.adv_attack import BaseAdvGenerator

class PGD(BaseAdvGenerator):
    def __init__(self, epsilon, n_step, step_size, rand_init, criterion_type="CrossEntropyLoss"):
        super(PGD, self).__init__(criterion_type)
        self.epsilon = epsilon
        self.rand_init = rand_init
        self.epsilon = epsilon
        self.n_step = n_step
        self.rand_init = rand_init
        self.step_size = step_size

    def generate_adv(self, net, inputs, targets, outputs=None):
        inputs_pgd = Variable(inputs.data.clone(), requires_grad=True)

        if self.rand_init:
            eta = inputs.new(inputs.size()).uniform_(-self.epsilon, self.epsilon)
            inputs_pgd.data = inputs_pgd + eta

        for _ in range(self.n_step):
            out = net(inputs_pgd)
            loss = self.criterion(out, Variable(targets))
            loss.backward()
            eta = self.step_size * inputs_pgd.grad.data.sign()
            inputs_pgd = Variable(inputs_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(inputs_pgd.data - inputs, -self.epsilon, self.epsilon)
            inputs_pgd.data = inputs + eta
        net.zero_grad()
        return inputs_pgd.data