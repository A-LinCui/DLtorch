import numpy as np
import random

import torch

def set_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)