from DLtorch.objective import *

Objective = {"BaseObjective": lambda: BaseObjective(),
             "ClassificationObjective": lambda **kwargs: ClassificationObjective(**kwargs),
             "ClassificationAdversarialObjective": lambda adversary_type, adversary_kwargs=None, adv_loss_coef=0.5,
                                                          adv_reward_coef=0.5, criterion_type="CrossEntropyLoss":
             ClassificationAdversarialObjective(adversary_type, adversary_kwargs, adv_loss_coef, adv_reward_coef, criterion_type)}

def get_objective(_type, **kwargs):
    assert _type in Objective.keys(), "NO Objective: ".format(_type)
    return Objective[_type](**kwargs)

def regist_objective(name, fun):
    Objective[name] = fun