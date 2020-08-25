# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

from DLtorch.objective import *

Objective = {"BaseObjective": lambda criterion_type="CrossEntropyLoss", criterion_kwargs=None: BaseObjective(criterion_type, criterion_kwargs),
             "ClassificationObjective": lambda **kwargs: ClassificationObjective(**kwargs),
             "ClassificationAdversarialObjective": lambda adversary_type, adversary_kwargs=None, adv_loss_coef=0.5,
                                                          adv_reward_coef=0.5, criterion_type="CrossEntropyLoss", criterion_kwargs=None:
             ClassificationAdversarialObjective(adversary_type, adversary_kwargs, adv_loss_coef, adv_reward_coef, criterion_type, criterion_kwargs)}

def get_objective(_type, **kwargs):
    # Get an objective from DLtorch framework.
    assert _type in Objective.keys(), "NO Objective: ".format(_type)
    return Objective[_type](**kwargs)

def regist_objective(name, fun):
    # Regist a new objective into DLtorch framework.
    Objective[name] = fun

def get_objective_attrs():
    # Get all the objective types.
    # Used in "main.components".
    return list(Objective.keys())