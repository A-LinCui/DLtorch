# -*- coding: utf-8 -*-

import abc
import os

import torchvision.transforms as transforms

from DLtorch.base import BaseComponent


class BaseDataset(BaseComponent):
    def __init__(self, path: str):
        super(BaseDataset, self).__init__()
        self.path = path
        self.datasets = {}
        self.logger.info("Load dataset from {}".format(os.path.abspath(self.path)))


class BaseCVDataset(BaseDataset):
    def __init__(
            self, 
            path: str, 
            train_transforms: dict = {},
            test_transforms: dict = {}
    ):
        super(BaseCVDataset, self).__init__(path)

        # Assemble transforms from dicts 
        self.train_transforms = transforms.Compose([getattr(transforms, _trans)(**train_transforms[_trans]) for _trans in train_transforms.keys()])
        self.test_transforms = transforms.Compose([getattr(transforms, _trans)(**test_transforms[_trans]) for _trans in test_transforms.keys()])

        self.logger.info("Train Transforms: {}".format(self.train_transforms))
        self.logger.info("Test Transforms: {}".format(self.test_transforms))
