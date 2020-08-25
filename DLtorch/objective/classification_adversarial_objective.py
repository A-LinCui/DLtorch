# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import torch

from DLtorch.utils import accuracy
from DLtorch.objective.base import BaseObjective
from DLtorch.adv_attack import get_attaker

class ClassificationAdversarialObjective(BaseObjective):
    NAME = "ClassificationAdversarialObjective"

    def __init__(self, adversary_type, adversary_kwargs=None,
                 adv_loss_coef=0.5, adv_reward_coef=0.5, criterion_type="CrossEntropyLoss", criterion_kwargs=None):
        super(ClassificationAdversarialObjective, self).__init__(criterion_type, criterion_kwargs)
        self.adversary_type = adversary_type
        self.adversary_kwargs = adversary_kwargs
        self.adversary = get_attaker(self.adversary_type, **self.adversary_kwargs)
        self.adv_loss_coef = adv_loss_coef
        self.adv_reward_coef = adv_reward_coef

    @ property
    def perf_names(self):
        return ["natrual_acc", "robust_acc"]

    def get_perfs(self, inputs, outputs, targets, model, **kwargs):
        if self.adv_loss_coef == 0:
            return [accuracy(outputs, targets)[0]]  # Top-1 accuracy
        else:
            # In training, we have generated adversarial examples while calculating the loss. Therefore, there's no need for us to generate again.
            if hasattr(self, "adversarial_examples"):
                perfs = [accuracy(outputs, targets)[0], accuracy(model(self.adversarial_examples), targets)[0]]
                del self.adversarial_examples
                return perfs
            else:
                adversarial_examples = self.adversary.generate_adv(model, inputs, targets, outputs)
                return [accuracy(outputs, targets)[0], accuracy(model(adversarial_examples), targets)[0]]

    def get_loss(self, inputs, outputs, targets, model):
        if self.adv_loss_coef == 0:
            return self._criterion(outputs, targets)
        else:
            self.adversarial_examples = self.adversary.generate_adv(model, inputs, targets, outputs)
            natural_loss = self._criterion(outputs, targets)
            adv_loss = self._criterion(model(self.adversarial_examples), targets)
            return (1 - self.adv_loss_coef) * natural_loss + self.adv_loss_coef * adv_loss

    def get_reward(self, perf):
        return (1 - self.adv_reward_coef) * perf[0] + self.adv_reward_coef * perf[1]