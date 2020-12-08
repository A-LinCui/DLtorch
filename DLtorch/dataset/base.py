# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import abc

from DLtorch.base import BaseComponent


class BaseDataset(BaseComponent):
    def __init__(self, path: str):
        super(BaseDataset, self).__init__(self)
        self.path = path
        self.datasets = {}

    @abc.abstractmethod
    def dataloader(self, **kwargs):
        """ Return the dataloaders. """