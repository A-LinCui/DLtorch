# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

from DLtorch.models import *

Models = {"Cifar_DenseNet121": lambda: DenseNet121(),
          "Cifar_DenseNet161": lambda: DenseNet161(),
          "Cifar_DenseNet169": lambda: DenseNet169(),
          "Cifar_DenseNet201": lambda: DenseNet201(),
          "Cifar_LeNet": lambda: LeNet(),
          "Cifar_resnet18": lambda: ResNet18(),
          "Cifar_resnet34": lambda: ResNet34(),
          "Cifar_resnet50": lambda: ResNet50(),
          "Cifar_resnet101": lambda: ResNet101(),
          "Cifar_resnet152": lambda: ResNet152(),
          "Cifar_WideResNet": lambda depth, num_classes, widen_factor, drop_rate: WideResNet(depth, num_classes, widen_factor, drop_rate),
          "MNIST_LeNet": lambda: MNIST_LeNet()
}

def get_model(_type, **kwargs):
    assert _type in Models.keys(), "NO Model: ".format(_type)
    return Models[_type](**kwargs)

def regist_model(name, fun):
    Models[name] = fun

def get_model_attrs():
    return list(Models.keys())