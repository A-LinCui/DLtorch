# Created by Junbo Zhao 2020.8.2
# File: DLtorch.config.py
# Explanation: Run "python.py" to generate global configuration.

import yaml
import os

from DLtorch.utils.python_utils import load_yaml
from DLtorch.utils.logger import logger

basic_path = os.path.dirname(__file__)

class config(object):
    def __init__(self):
        self.basic_path = basic_path
        self.dir = os.path.abspath(os.path.join(basic_path, "config.yaml"))
        log = logger("DLtorch Config")
        if os.path.exists(self.dir):
            self.configuration = load_yaml(self.dir)
        else:
            self.init()

    def __call__(self):
        return self.configuration

    @ property
    def datasets(self):
        return self.configuration["Dataset"]

    def init(self):
        self.config_all = {}
        # Set the datasets
        self.dataset = {}
        self.dataset["Cifar10"] = os.path.abspath(os.path.join(self.basic_path, "./datasets/data/cifar10"))
        self.dataset["MNIST"] = os.path.abspath(os.path.join(self.basic_path, "./datasets/data/MNIST"))
        self.dataset["FashionMNIST"] = os.path.abspath(os.path.join(self.basic_path, "./datasets/data/FashionMNIST"))
        self.dataset["Cifar100"] = os.path.abspath(os.path.join(self.basic_path, "./datasets/data/cifar100"))
        self.config_all["Dataset"] = self.dataset
        with open(self.dir, "w", encoding="utf-8") as f:
            yaml.dump(self.config_all, f)
        self.configuration = self.config_all

    # --- Helper ---
    def rewrite_config(self):
        with open(self.dir, "w", encoding="utf-8") as f:
            yaml.dump(self.configuration, f)

if __name__ == "__main__":
    config()