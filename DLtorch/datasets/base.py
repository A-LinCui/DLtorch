import torch.utils.data as data
from DLtorch.config import config

class base_dataset(object):
    def __init__(self, datatype, whether_valid):
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

    def dataloader(self, batch_size, num_workers=0, train_shuffle=True, test_shuffle=False, drop_last=False):
        dataloader = {
            "train": data.DataLoader(dataset=self.datasets["train"], batch_size=batch_size, num_workers=num_workers,
                                     shuffle=train_shuffle, drop_last=drop_last),
            "test": data.DataLoader(dataset=self.datasets["test"], batch_size=batch_size, num_workers=num_workers,
                                    shuffle=test_shuffle, drop_last=drop_last)}
        if self.whether_valid:
            dataloader["valid"] = data.DataLoader(dataset=self.datasets["valid"], batch_size=batch_size, num_workers=num_workers,
                                  shuffle=test_shuffle, drop_last=drop_last)
        return dataloader