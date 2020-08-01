import yaml
import os

basic_path = os.path.dirname(__file__)

class config(object):
    def __init__(self, whether_init=False):
        self.basic_path = basic_path
        self.dir = os.path.join(basic_path, "config.yaml")
        file = open(self.dir, "r")
        self.configuration = yaml.load(file, Loader=yaml.FullLoader)

        # Initialize the configuration file
        if whether_init:
            self.config_all = {}

            # Set the datasets
            self.dataset = {}
            self.dataset["Cifar10"] = os.path.abspath(os.path.join(self.basic_path, "./datasets/data/cifar10"))
            self.dataset["MNIST"] = os.path.abspath(os.path.join(self.basic_path, "./datasets/data/MNIST"))
            self.dataset["FashionMNIST"] = os.path.abspath(os.path.join(self.basic_path, "./datasets/data/FashionMNIST"))
            self.config_all["Dataset"] = self.dataset
            with open(self.dir, "w", encoding="utf-8") as f:
                yaml.dump(self.config_all, f)

    def __call__(self):
        return self.configuration

    @ property
    def datasets(self):
        return self.configuration["Dataset"]

if __name__ == "__main__":
    config(whether_init=True)