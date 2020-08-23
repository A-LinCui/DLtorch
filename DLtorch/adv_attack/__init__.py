from DLtorch.adv_attack.base import BaseAdvGenerator
from DLtorch.adv_attack.FGSM import FGSM
from DLtorch.adv_attack.PGD import PGD

Attacker = {"FGSM": lambda epsilon, rand_init, criterion_type="CrossEntropyLoss": FGSM(epsilon, rand_init, criterion_type),
            "PGD": lambda epsilon, n_step, step_size, rand_init, criterion_type="CrossEntropyLoss": PGD(epsilon, n_step, step_size, rand_init, criterion_type)}

def get_attaker(_type, **kwargs):
    assert _type in Attacker.keys(), "NO Attacker: ".format(_type)
    return Attacker[_type](**kwargs)

def regist_attacker(name, fun):
    Attacker[name] = fun

def get_attacker_attrs():
    return list(Attacker.keys())