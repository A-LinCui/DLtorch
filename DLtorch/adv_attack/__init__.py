# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

from DLtorch.adv_attack.base import BaseAdvGenerator
from DLtorch.adv_attack.FGSM import FGSM
from DLtorch.adv_attack.PGD import PGD

Attacker = {"FGSM": lambda **kwargs: FGSM(**kwargs),
            "PGD": lambda **kwargs: PGD(**kwargs)}

def get_attaker(_type, **kwargs):
    # Get an attacker from DLtorch framework.
    assert _type in Attacker.keys(), "NO Attacker: {}".format(_type)
    return Attacker[_type](**kwargs)

def regist_attacker(name, fun):
    # Regist an attacker into DLtorch framework.
    Attacker[name] = fun

def get_attacker_attrs():
    # Get all the attacker types.
    # Used in "main.components".
    return list(Attacker.keys())