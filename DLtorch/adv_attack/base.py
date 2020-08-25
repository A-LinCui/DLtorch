# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import abc

from DLtorch.component.criterion import get_criterion

class BaseAdvGenerator(object):
    def __init__(self, criterion_type="CrossEntropyLoss", criterion_kwargs=None):
        self.criterion = get_criterion(criterion_type, **criterion_kwargs) if criterion_kwargs is not None \
            else get_criterion(criterion_type)

    @abc.abstractmethod
    def generate_adv(self, net, inputs, targets, outputs):
        """
        Generate adversarial types of the inputs.
        """