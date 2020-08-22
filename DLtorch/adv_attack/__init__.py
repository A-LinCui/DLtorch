from DLtorch.adv_attack.base import BaseAdvGenerator
from DLtorch.adv_attack.FGSM import FGSM

Attacker = {"FGSM": lambda epsilon, rand_init, criterion_type: FGSM(epsilon, rand_init, criterion_type)}

def get_attaker(_type, **kwargs):
    assert _type in Attacker.keys(), "NO Attacker: ".format(_type)
    return Attacker[_type](**kwargs)

def regist_attacker(name, fun):
    Attacker[name] = fun