import abc

import torch.nn as nn

class BaseCriterion(nn.Module):
    def __init__(self):
        super().__init__()