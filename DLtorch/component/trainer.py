from DLtorch.train import *

Trainer = {"FinalTrainer": lambda **kwargs: FinalTrainer(**kwargs)}

def get_trainer(_type, **kwargs):
    assert _type in Trainer.keys(), "NO Trainer: ".format(_type)
    return Trainer[_type](**kwargs)

def regist_trainer(name, fun):
    Trainer[name] = fun