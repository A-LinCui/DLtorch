# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from DLtorch.dataset.base import BaseDataset

class Cifar10(BaseDataset):
    NAME = "cifar10"

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
        },
        valid: bool = True, 
        portion: list[float] = None):

        super(Cifar10, self).__init__(path)
        self.train_transforms = transforms.Compose([])
        for _trans in train_transforms.keys():
            self.train_transforms.append(getattr(transforms, _trans)(**train_transforms[_trans]))
        
        self.test_transforms = transforms.Compose([])
        for _trans in test_transforms.keys():
            self.test_transforms.append(getattr(transforms, _trans)(**test_transforms[_trans]))

        self.datasets["train"] = datasets.CIFAR10(root=self.path, train=True, download=True, transform=self.train_transform)
        self.datasets["test"] = datasets.CIFAR10(root=self.path, train=False, download=True, transform=self.test_transform)
        self.datalength["train"] = len(self.datasets["train"])
        self.datalength["test"] = len(self.datasets["test"])
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')