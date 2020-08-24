import torchvision.transforms as transforms
import torchvision.datasets as datasets

from DLtorch.datasets.base import base_dataset

class SVHN(base_dataset):
    NAME = "SVHN"

    def __init__(self, dir, mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970],
                 train_transform=None, test_transform=None, extra_transform=None, whether_valid=False, portion=None):
        super(SVHN, self).__init__(datatype="image", whether_valid=whether_valid, portion=portion)
        self.dir = dir
        self.mean = mean
        self.std = std
        self.train_transform = train_transform if train_transform is not None else \
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
        self.test_transform = test_transform if test_transform is not None else \
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
        self.extra_transform = extra_transform if extra_transform is not None else \
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])

        self.datasets["train"] = datasets.SVHN(root=self.dir, split="train", download=True, transform=self.train_transform)
        self.datasets["test"] = datasets.SVHN(root=self.dir, split="test", download=True, transform=self.test_transform)
        self.datasets["extra"] = datasets.SVHN(root=self.dir, split="extra", download=True, transform=self.extra_transform)
        self.datalength["train"] = len(self.datasets["train"])
        self.datalength["test"] = len(self.datasets["test"])
        self.datalength["extra"] = len(self.datasets["extra"])

        if self.whether_valid:
            self.devide()