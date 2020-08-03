from DLtorch.models import *

Models = {"Cifar10_DenseNet121": lambda: DenseNet121(),
          "Cifar10_DenseNet161": lambda: DenseNet161(),
          "Cifar10_DenseNet169": lambda: DenseNet169(),
          "Cifar10_DenseNet201": lambda: DenseNet201(),
          "Cifar10_LeNet": lambda: LeNet(),
          "Cifar10_resnet18": lambda: ResNet18(),
          "Cifar10_resnet34": lambda: ResNet34(),
          "Cifar10_resnet50": lambda: ResNet50(),
          "Cifar10_resnet101": lambda: ResNet101(),
          "Cifar10_resnet152": lambda: ResNet152(),
          "Cifar10_WideResNet": lambda depth, num_classes, widen_factor, drop_rate: WideResNet(depth, num_classes, widen_factor, drop_rate)
}

def get_model(_type, **kwargs):
    assert _type in Models.keys(), "NO Model: ".format(_type)
    return Models[_type](**kwargs)

def regist_model(name, fun):
    Models[name] = fun