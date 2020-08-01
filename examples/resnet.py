import os
import argparse

import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import DataParallel
import torch.nn as nn

from DLtorch.datasets import Cifar10
from DLtorch.models import WideResNet
import DLtorch.utils as utils
import DLtorch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="resnet_train_config.yaml", help="config file")
    parser.add_argument('--gpus', type=str, default="0", help="gpus")
    args = parser.parse_args()

    # Set the logger
    log = utils.logger(name="train", save_path=None, whether_stream=True, whether_file=False)
    log.info("Using DLtorch Framework")
    # Load the config file
    config = utils.load_yaml(args.config)
    # Set used gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    # Load the data
    dataloader = Cifar10().dataloader(**config["Dataset"]["Dataloader"])
    log.info("Load data {} from {}.".format(config["Dataset"]["type"], DLtorch.config.config().datasets[config["Dataset"]["type"]]))
    # Load the models
    model = WideResNet(10, 10).cuda()
    if len(str(args.gpus)) > 1:
        model = DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    params = list(model.parameters())
    optimizer = optim.SGD(params, lr=float(config["Train"]["lr"]), weight_decay=float(config["Train"]["weight_decay"]), momentum=float(config["Train"]["momentum"]), nesterov=True)
    scheduler = MultiStepLR(optimizer, milestones=[int(config["Train"]["epochs"] / 2),  int(config["Train"]["epochs"] * 3 / 4),
                                                   int(config["Train"]["epochs"] * 7 / 8)], gamma=float(config["Train"]["scheduler"]["gamma"]))

    for (images, labels) in dataloader["train"]:
        print(images)