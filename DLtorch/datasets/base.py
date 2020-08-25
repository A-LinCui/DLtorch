# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import torch.utils.data as data
import torch

class base_dataset(object):
    def __init__(self, whether_valid, portion, datatype):
        self.datatype = datatype
        self.whether_valid = whether_valid
        self.portion = portion
        self.datasets = {}
        self.datalength = {}

    @property
    def get_datatype(self):
        return self.datatype

    @property
    def get_datalength(self):
        return self.datalength

    @ property
    def dataset(self):
        return self.datasets

    def dataloader(self, **kwargs):
        dataloader = {
            "train": data.DataLoader(dataset=self.datasets["train"], **kwargs["trainset"]),
            "test": data.DataLoader(dataset=self.datasets["test"], **kwargs["testset"])}
        if self.whether_valid:
            dataloader["valid"] = data.DataLoader(dataset=self.datasets["valid"], **kwargs["testset"])
        return dataloader

    def devide(self):
        assert self.portion is not None, "Data portion is needed if using validation set."
        assert sum(self.portion) == 1.0, "Data portion invalid. The sum of training set and validation set should be 1.0"
        self.datalength["valid"] = int(self.portion[1] * self.datalength["train"])
        self.datalength["train"] = self.datalength["train"] - self.datalength["valid"]
        self.datasets["train"], self.datasets["valid"] = torch.utils.data.random_split(
            self.datasets["train"], [self.datalength["train"], self.datalength["valid"]])
        self.datasets["valid"].transform = self.test_transform