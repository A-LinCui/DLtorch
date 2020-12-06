# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import numpy as np
import random
from collections import OrderedDict

import torch

def _set_seed(seed):
    """ Set seed for system, numpy and pytorch. """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def is_empty(self):
        return self.cnt == 0

class EnsembleAverageMeters(object):
    def __init__(self):
        self.AverageMeters = None

    def is_empty(self):
        return self.AverageMeters is None

    def update(self, perfs, n=1):
        if self.is_empty():
            self.AverageMeters = OrderedDict([(name, AvgrageMeter()) for name in perfs])
        [self.AverageMeters[name].update(val, n) for name, val in perfs.items()]

    def avgs(self):
        return OrderedDict((name, val.avg) for name, val in self.AverageMeters.items()) if not self.is_empty() else None

    def items(self):
        return self.AverageMeters.items() if not self.is_empty() else None

    def reset(self):
        self.AverageMeters = None