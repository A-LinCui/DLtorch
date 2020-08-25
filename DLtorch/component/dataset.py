# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

from DLtorch.datasets import *

datasets = {
    "Cifar10": lambda **kwargs: Cifar10(**kwargs),
    "Cifar100": lambda **kwargs: Cifar100(**kwargs),
    "FashionMNIST": lambda **kwargs: FashionMNIST(**kwargs),
    "MNIST": lambda **kwargs: MNIST(**kwargs),
    "ImageNet": lambda **kwargs: Imagenet(**kwargs),
    "SVHN": lambda **kwargs: SVHN(**kwargs)
}

def get_dataset(_type, **kwargs):
    # Get a criterion from DLtorch framework.
    assert _type in datasets.keys(), "No dataset: {}".format(_type)
    return datasets[_type](**kwargs)

def get_dataset_attr():
    # Get all the dataset types.
    # Used in "main.components".
    return list(datasets.keys())

def regist_dataset(name, fun):
    # Regist a dataset into DLtorch Framework.
    datasets[name] = fun