# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import abc

import torch

class BaseAdvGenerator(object):
    def __init__(self, criterion_type="CrossEntropyLoss"):
        self.criterion = getattr(torch.nn, criterion_type)()

    @abc.abstractmethod
    def generate_adv(self, net, inputs, targets, outputs):
        """
        Generate adversarial types of the inputs.
        """