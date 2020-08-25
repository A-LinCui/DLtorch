# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from DLtorch.datasets.base import base_dataset

class Imagenet(base_dataset):
    NAME = "Imagenet"

    def __init__(self, train_dir, test_dir, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                 train_transform=None, test_transform=None, whether_valid=False, portion=None):
        super(Imagenet, self).__init__(datatype="image", whether_valid=whether_valid, portion=portion)
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.mean = mean
        self.std = std
        self.train_transform = train_transform if train_transform is not None else \
            transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self,std),
            ])
        self.test_transform = test_transform if test_transform is not None else \
            transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
        self.datasets["train"] = datasets.ImageFolder(root=self.train_dir, transform=self.train_transform)
        self.datasets["test"] = datasets.ImageFolder(root=self.test_dir, transform=self.test_transform)
        self.datalength["train"] = len(self.datasets["train"])
        self.datalength["test"] = len(self.datasets["test"])
        if self.whether_valid:
            self.devide()