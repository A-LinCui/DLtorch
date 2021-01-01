# -*- coding:utf-8 -*-

import torchvision.datasets as datasets

from DLtorch.dataset.base import BaseCVDataset


class ImageNet(BaseCVDataset):
    def __init__(
        self,
        train_path: str,
        test_path: str,
        train_transforms: dict = {
            "RandomSizedCrop": {"size": 224},
            "RandomHorizontalFlip": {},
            "ToTensor": {},
            "Normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                }
        },
        test_transforms: dict = {
            "Scale": {"size": 256},
            "CenterCrop": {"size": 224},
            "ToTensor": {}, 
            "Normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                }
        }
        ):
        super(ImageNet, self).__init__(train_path, train_transforms, test_transforms)

        self.train_path = train_path
        self.test_path = test_path

        # Load the dataset
        self.datasets["train"] = datasets.ImageFolder(root=self.train_path, transform=self.train_transforms)
        self.datasets["test"] = datasets.ImageFolder(root=self.test_path, transform=self.test_transforms)
