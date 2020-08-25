# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

from DLtorch.train import *

Trainer = {"CNNFinalTrainer": lambda **kwargs: CNNFinalTrainer(**kwargs)}

def get_trainer(_type, **kwargs):
    # Get a train from DLtorch framework.
    assert _type in Trainer.keys(), "NO Trainer: ".format(_type)
    return Trainer[_type](**kwargs)

def regist_trainer(name, fun):
    # Regist a new trainer into DLtorch framework.
    Trainer[name] = fun

def get_trainer_attrs():
    # Get all the trainer types.
    # Used in "main.components".
    return list(Trainer.keys())