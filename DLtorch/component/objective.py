import DLtorch.objective

def get_objective_cls(_type):
    return getattr(DLtorch.objective, _type)

def get_objective(_type, **kwargs):
    return get_objective_cls(_type)(**kwargs)