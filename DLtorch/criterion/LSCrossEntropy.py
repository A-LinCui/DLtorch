# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import torch
import torch.nn as nn

from DLtorch.criterion.base import BaseCriterion

class LSCrossEntropy(BaseCriterion):
    """ CrossEntropy with Label Smoothing. """
    
    def __init__(self, smooth: float = 0.2):
        super(LSCrossEntropy, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        num_classes = int(inputs.shape[-1])
        log_probs = nn.LogSoftmax(dim=1)(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.smooth) * targets + self.smooth / num_classes
        loss = - (targets * log_probs).mean(0).sum()
        return loss