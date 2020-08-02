import DLtorch.datasets

def get_dataset_cls(_type):
    return getattr(DLtorch.datasets, _type)

def get_dataset(_type, **kwargs):
    return get_dataset_cls(_type)(**kwargs)