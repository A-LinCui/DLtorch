# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from DLtorch.datasets.base import base_dataset

class Cifar10(base_dataset):
    NAME = "cifar10"

    def __init__(self, dir, mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768],
                 train_transform=None, test_transform=None, whether_valid=False, portion=None):
        super(Cifar10, self).__init__(datatype="image", whether_valid=whether_valid, portion=portion)
        self.dir = dir
        self.mean = mean
        self.std = std
        self.train_transform = train_transform if train_transform is not None else \
            transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
        self.test_transform = test_transform if test_transform is not None else \
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])

        self.datasets["train"] = datasets.CIFAR10(root=self.dir, train=True, download=True, transform=self.train_transform)
        self.datasets["test"] = datasets.CIFAR10(root=self.dir, train=False, download=True, transform=self.test_transform)
        self.datalength["train"] = len(self.datasets["train"])
        self.datalength["test"] = len(self.datasets["test"])
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        if self.whether_valid:
            self.devide()