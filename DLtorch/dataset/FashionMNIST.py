# -*- coding:utf-8 -*-

import torchvision.datasets as datasets

from DLtorch.dataset.base import BaseCVDataset


class FashionMNIST(BaseCVDataset):
    def __init__(
        self, 
        path: str, 
        train_transforms: dict = {
            "ToTensor": {}
            },
        test_transforms: dict = {
            "ToTensor": {}
            }
        ):
        super(FashionMNIST, self).__init__(path, train_transforms, test_transforms)
        
        # Load the dataset
        self.datasets["train"] = datasets.FashionMNIST(root=self.path, train=True, download=True, transform=self.train_transforms)
        self.datasets["test"] = datasets.FashionMNIST(root=self.path, train=False, download=True, transform=self.test_transforms)
