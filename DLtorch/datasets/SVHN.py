# -*- coding:utf-8 -*-

import torchvision.datasets as datasets

from DLtorch.datasets.base import BaseCVDataset


class SVHN(BaseCVDataset):
    def __init__(
        self, 
        path: str, 
        train_transforms: dict = {
            "ToTensor": {},
            "Normalize": {
                "mean": [0.4377, 0.4438, 0.4728],
                "std": [0.1980, 0.2010, 0.1970]
                }
            }
         test_transforms: dict = {
            "ToTensor": {},
            "Normalize": {
                "mean": [0.4377, 0.4438, 0.4728],
                "std": [0.1980, 0.2010, 0.1970]
                }
            }
         extra_transforms: dict = None
        ):
        super(SVHN, self).__init__(path, train_transforms, test_transforms)

        if extra_transforms is None:
            self.extra_transforms = self.test_transforms
            self.logger.info("Extra transforms not given. Automatically use test transforms as extra transforms")
        else:
            self.extra_transforms = extra_transforms
            self.logger.info("Extra Transforms: {}".format(self.extra_transforms))
        
        # Load the dataset
        self.datasets["train"] = datasets.SVHN(root=self.path, split="train", download=True, transform=self.train_transforms)
        self.datasets["test"] = datasets.SVHN(root=self.path, split="test", download=True, transform=self.test_transforms)
        self.datasets["extra"] = datasets.SVHN(root=self.dir, split="extra", download=True, transform=self.extra_transforms)
