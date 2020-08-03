from DLtorch.objective import *

Objective = {"BaseObjective": lambda: BaseObjective(),
             "ClassificationObjective": lambda **kwargs: ClassificationObjective(**kwargs)}

def get_objective(_type, **kwargs):
    assert _type in Objective.keys(), "NO Objective: ".format(_type)
    return Objective[_type](**kwargs)

def regist_objective(name, fun):
    Objective[name] = fun