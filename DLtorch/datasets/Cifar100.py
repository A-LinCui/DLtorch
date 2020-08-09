import torchvision.transforms as transforms
import torchvision.datasets as datasets

from DLtorch.datasets.base import base_dataset

class Cifar100(base_dataset):
    NAME = "Cifar100"

    def __init__(self, mean=[0.5070751592371322, 0.4865488733149497, 0.44091784336703466],
                 std=[0.26733428587924063, 0.25643846291708833, 0.27615047132568393],
                 train_transform=None, test_transform=None,
                 whether_valid=False, portion=None):
        super(Cifar100, self).__init__(mean=mean, std=std, datatype="image", whether_valid=whether_valid, portion=portion)

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

        self.datasets["train"] = datasets.CIFAR100(root=self.datasets_dir["Cifar100"], train=True, download=True, transform=self.train_transform)
        self.datasets["test"] = datasets.CIFAR100(root=self.datasets_dir["Cifar100"], train=False, download=True, transform=self.test_transform)
        self.datalength["train"] = len(self.datasets["train"])
        self.datalength["test"] = len(self.datasets["test"])

        if self.whether_valid:
            self.devide()