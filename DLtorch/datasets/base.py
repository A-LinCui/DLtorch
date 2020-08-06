import torch.utils.data as data
from DLtorch.config import config

class base_dataset(object):
    def __init__(self, datatype=None, whether_valid=False):
        self.datatype = datatype
        self.whether_valid = whether_valid
        self.datasets = {}
        self.datalength = {}
        self.configuration = config()
        self.datasets_dir = self.configuration.datasets

    @property
    def get_datatype(self):
        return self.datatype

    @property
    def get_datalength(self):
        return self.datalength

    @ property
    def dataset(self):
        return self.datasets

    # def dataloader(self, batch_size, num_workers=0, train_shuffle=True, test_shuffle=False, drop_last=False):
    def dataloader(self, **kwargs):
        dataloader = {
            "train": data.DataLoader(dataset=self.datasets["train"], **kwargs["trainset"]),
            "test": data.DataLoader(dataset=self.datasets["test"], **kwargs["testset"])}
        if self.whether_valid:
            dataloader["valid"] = data.DataLoader(dataset=self.datasets["valid"], **kwargs["testset"])
        return dataloader