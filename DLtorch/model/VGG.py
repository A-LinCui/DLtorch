# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from DLtorch.model.base import BaseModel

"""
Very Deep Convolutional Networks for Large-Scale Image Recognition.
Karen Simonyan, Andrew Zisserman, https://arxiv.org/abs/1409.1556v6
"""


def make_layers(cfg, batch_norm=False):
    layers = []
    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue
        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    return nn.Sequential(*layers)


class BaseCifarVGG(nn.Module):
    def __init__(
        self, 
        features: list, 
        num_class: int = 10
        ):
        super(BaseCifarVGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output


class CifarVGG(BaseModel, BaseCifarVGG):
    def __init__(
        self,
        feature: list,
        num_classes: int = 10
        ):
        BaseCifarVGG.__init__(self, feature, num_classes)
        BaseModel.__init__(self)


"""
VGG11: model_kwargs: dict = {
            "feature": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        }
VGG13: model_kwargs: dict = {
            "feature": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        }
VGG16: model_kwargs: dict = {
            "feature": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        }
VGG19: model_kwargs: dict = {
            "feature": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        }
"""