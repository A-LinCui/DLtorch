import DLtorch.models

def get_model_cls(type_):
    return getattr(DLtorch.models, type_)

def get_model(_type, **kwargs):
    return get_model_cls(_type)(**kwargs)