# -*- coding:utf-8 -*-

from __future__ import print_function

import os
import functools
import click
import yaml
import importlib
import sys

import torch

import DLtorch
from DLtorch.base import Plugins
from DLtorch.utils import logger as _logger
from DLtorch.utils.common_utils import _set_seed
from DLtorch.version import __version__


install_path = os.path.dirname(os.path.abspath(__file__))
root_file = os.path.join(install_path, "root.yaml")


def _set_gpu(gpu):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        LOGGER.info("GPU device = {}".format(gpu))
    else:
        LOGGER.warning('No GPU available, use CPU!!')


# Set the logger
LOGGER = _logger.getChild("Main")


# Check plugins
if not os.path.exists(root_file):
    plugin_root = os.path.join("~", "dltorch_plugins")
    with open(root_file, "w") as f:
        yaml.dump(os.path.abspath(plugin_root), f)
    if not os.path.exists(plugin_root):
        os.makedirs(plugin_root)
    LOGGER.info("Initialize DLtorch with default plugins root: {}".format(plugin_root))
    LOGGER.info("All modules under the plugins root that is subclass of 'DLtorch.base.BaseComponent' will be automatically loaded.")
    LOGGER.info("Able to change the plugins root with 'DLtorch setroot'.")
else:
    with open(root_file, "r") as f:
        plugin_root = yaml.load(f, Loader=yaml.FullLoader)

LOGGER.info("Check plugins under {}".format(os.path.abspath(plugin_root)))
DLTORCH_PlUGINS = Plugins(plugin_root=plugin_root)    


click.option = functools.partial(click.option, show_default=True)

@click.group(help="The DLtorch framework command line interface.")
@click.version_option(version=__version__)
@click.option("--local_rank", default=-1, type=int, help="The rank of this process")
def main(local_rank):
    pass


@click.command(help="Set Plugins Root")
@click.argument("root", required=True, type=str)
def setroot(root):
    root = os.path.abspath(root)
    assert(os.path.exists(root)), "Root {} doesn't exist!".format(root)
    with open(root_file, "w") as f:
        yaml.dump(root, f)
    LOGGER.info("Successful set plugin root: {}".format(root))

main.add_command(setroot)


@click.command(help="Train Model")
@click.argument("cfg_file", required=True, type=str)
@click.option("--seed", default=123, type=int, help="The random seed to run training")
@click.option("--load", default=None, type=str, help="The directory to load checkpoint")
@click.option("--train-dir", default=None, type=str, help="The directory to save checkpoints")
@click.option('--gpus', default="0", type=str, help="GPUs to use")
@click.option('--save-every', default=None, type=int, help="Number of epochs to save once")
@click.option('--report-every', default=50, type=int, help="Number of batches to report once in a epoch")
def train(cfg_file, seed, load, train_dir, gpus, save_every, report_every):
    # Set the device
    gpu_list = [int(gpu) for gpu in gpus.split(",")]
    if not gpu_list:
        device = torch.device("cpu")
        LOGGER.info("Current Device: CPU")
    else:
        _set_gpu(gpu_list[0])
        device = torch.device("cuda:{}".format(gpu_list[0]) if torch.cuda.is_available() else "cpu")
    
    # Set the seed
    if seed is not None:
        LOGGER.info("Setting random seed: {}".format(seed))
        _set_seed(seed)
    
    # Load the configuration
    with open(cfg_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Makedir
    os.makedirs(train_dir)
    with open(os.path.join(train_dir, "train_config.yaml"), "w") as f:
        yaml.dump(config, f)
    log_file = os.path.join(train_dir, "train.log")
    _logger.addFile(log_file)

    # Init components
    model = getattr(DLtorch.model, config["model_type"])(**config["model_kwargs"]).to(device)
    if device != torch.device("cpu") and len(gpu_list) > 1:
        model = torch.nn.DataParallel(model)
    objective = getattr(DLtorch.objective, config["objective_type"])(**config["objective_kwargs"])
    dataset = getattr(DLtorch.datasets, config["dataset_type"])(**config["dataset_kwargs"])
    trainer = getattr(DLtorch.trainer, config["trainer_type"])(
        **config["trainer_kwargs"], device=device, gpu_list=gpu_list, save_every=save_every, report_every=report_every,
        model=model, dataset=dataset, objective=objective)
    
    if load is not None:
        trainer.load(load)
    trainer.train()
    
    return trainer

main.add_command(train)


@click.command(help="Test Model")
@click.argument("cfg_file", required=True, type=str)
@click.option("--split", required=True, type=click.Choice(['train', 'test']), help="Dataset split to test")
@click.option("--seed", default=123, type=int, help="The random seed to run training")
@click.option("--load", required=True, type=str, help="The directory to load checkpoint")
@click.option('--gpus', default="0", type=str, help="GPUs to use")
@click.option('--report-every', type=int, default=50, help="Number of batches to report once in a epoch")
def test(cfg_file, split, seed, load, gpus, report_every):
    # Set the device
    gpu_list = [int(gpu) for gpu in gpus.split(",")]
    if not gpu_list:
        device = torch.device("cpu")
        LOGGER.info("Current Device: CPU")
    else:
        _set_gpu(gpu_list[0])
        device = torch.device("cuda:{}".format(gpu_list[0]) if torch.cuda.is_available() else "cpu")
    
    # Set the seed
    if seed is not None:
        LOGGER.info("Setting random seed: {}".format(seed))
        _set_seed(seed)
    
    # Load the configuration
    with open(cfg_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Init components
    objective = getattr(DLtorch.objective, config["objective_type"])(**config["objective_kwargs"])
    model = getattr(DLtorch.model, config["model_type"])(**config["model_kwargs"])
    dataset = getattr(DLtorch.datasets, config["dataset_type"])(**config["dataset_kwargs"])
    trainer = getattr(DLtorch.trainer, config["trainer_type"])(
        **config["trainer_kwargs"], device=device, gpu_list=gpu_list, report_every=report_every,
        model=model, dataset=dataset, objective=objective)
    
    trainer.load(load)
    trainer.test(split)
    
    return trainer

main.add_command(test)


if __name__ == '__main__':
    main()