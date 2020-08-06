import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from DLtorch.datasets.base import base_dataset

class Cifar10(base_dataset):
    NAME = "Cifar10"

    def __init__(self, mean=None, std=None, train_transform=None, test_transform=None, whether_valid=False, portion=None):
        super(Cifar10, self).__init__(datatype="image", whether_valid=whether_valid)
        self.mean = mean if mean is not None else [0.49139968, 0.48215827, 0.44653124]
        self.std = std if std is not None else [0.24703233, 0.24348505, 0.26158768]

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

        self.datasets["train"] = datasets.CIFAR10(root=self.datasets_dir["Cifar10"], train=True, download=True, transform=self.train_transform)
        self.datasets["test"] = datasets.CIFAR10(root=self.datasets_dir["Cifar10"], train=False, download=True, transform=self.test_transform)
        self.datalength["train"] = len(self.datasets["train"])
        self.datalength["test"] = len(self.datasets["test"])

        if self.whether_valid:
            assert portion is not None, "Data portion is needed if using validation set."
            assert sum(portion) == 1.0, "Data portion invalid. The sum of training set and validation set should be 1.0"
            self.datalength["valid"] = int(portion[1] * self.datalength["train"])
            self.datalength["train"] = self.datalength["train"] - self.datalength["valid"]
            self.datasets["train"], self.datasets["valid"] = torch.utils.data.random_split(
                self.datasets["train"], [self.datalength["train"], self.datalength["valid"]])
            self.datasets["valid"].transform = self.test_transform

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')