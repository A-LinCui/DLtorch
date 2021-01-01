# -*- coding:utf-8 -*-

import torchvision.datasets as datasets

from DLtorch.dataset.base import BaseCVDataset


class Cifar100(base_dataset):
    def __init__(
        self, 
        path: str, 
        train_transforms: dict = {
            "RandomCrop": {"size": 32, "padding": 4},
            "RandomHorizontalFlip": {},
            "ToTensor": {},
            "Normalize": {
                "mean": [0.5070751592371322, 0.4865488733149497, 0.44091784336703466],
                "std": [0.26733428587924063, 0.25643846291708833, 0.27615047132568393]
                }
            },
         test_transforms: dict = {
            "ToTensor": {},
            "Normalize": {
                "mean": [0.5070751592371322, 0.4865488733149497, 0.44091784336703466],
                "std": [0.26733428587924063, 0.25643846291708833, 0.27615047132568393]
                }
            }
        ):
        super(Cifar100, self).__init__(path, train_transforms, test_transforms)
        
        # Load the dataset
        self.datasets["train"] = datasets.CIFAR100(root=self.path, train=True, download=True, transform=self.train_transforms)
        self.datasets["test"] = datasets.CIFAR100(root=self.path, train=False, download=True, transform=self.test_transforms)
