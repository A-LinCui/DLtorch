# -*- coding: utf-8 -*-

import torchvision.datasets as datasets

from DLtorch.dataset.base import BaseCVDataset


class Cifar10(BaseCVDataset):
    def __init__(
        self, 
        path: str, 
        train_transforms: dict = {
            "RandomCrop": {"size": 32, "padding": 4},
            "RandomHorizontalFlip": {},
            "ToTensor": {},
            "Normalize": {
                "mean": [0.49139968, 0.48215827, 0.44653124], 
                "std": [0.24703233, 0.24348505, 0.26158768]
                }
        }, 
        test_transforms: dict = {
            "ToTensor": {},
            "Normalize": {
                "mean": [0.49139968, 0.48215827, 0.44653124], 
                "std": [0.24703233, 0.24348505, 0.26158768]
                }
            }
        ):
        super(Cifar10, self).__init__(path, train_transforms, test_tranforms)
        
        # Load the dataset
        self.datasets["train"] = datasets.CIFAR10(root=self.path, train=True, download=True, transform=self.train_transforms)
        self.datasets["test"] = datasets.CIFAR10(root=self.path, train=False, download=True, transform=self.test_transforms)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
