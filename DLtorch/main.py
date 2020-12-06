# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

from __future__ import print_function

import os
import functools
import click
import importlib
import sys

from DLtorch.utils import logger as _logger
from DLtorch.component import *
from DLtorch.adv_attack import *
from DLtorch.utils import *
from DLtorch.version import __version__

def _set_gpu(gpu):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        LOGGER.info('GPU device = %d' % gpu)
    else:
        LOGGER.warning('No GPU available, use CPU!!')

def register(path):
    # Automatically run the "register" function defined in the "path(.py)" file to register new components into DLtorch.
    p, f = os.path.split(os.path.abspath(path))
    sys.path.append(p)
    module = importlib.import_module(f[:-3])
    module.register()
    sys.path.remove(p)

def register_components(path):
    # Register all the given files.
    if isinstance(path, str):
        register(path)
    else:
        [register(one_path) for one_path in path]

def prepare(config, device, dir, gpus, save_every):
    # Prepare working folder for the process and modify the configuration.
    if device is not None:
        assert device in ["cuda", "cpu"], "Device should be 'cuda' or 'cpu'."
        config["device"] = device
    if dir is not None:
        config["path"] = dir
    else:
        config["path"] = None
    if gpus is not None:
        config["gpus"] = gpus
    if save_every is not None:
        config["save_every"] = save_every
    if dir is not None and not os.path.exists(dir):
        os.mkdir(dir)
    return config


click.option = functools.partial(click.option, show_default=True)

# Set the logger
LOGGER = _logger.getChild("Main")
# Set the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.group(help="The DLtorch framework command line interface.")
@click.version_option(version=__version__)
@click.option("--local_rank", default=-1, type=int, help="The rank of this process")
def main(local_rank):
    pass

@click.command(help="Train Model")
@click.argument("cfg_file", required=True, type=str)
@click.option("--seed", default=100, type=int, help="The random seed to run training")
@click.option("--load", default=None, type=str, help="The directory to load checkpoint")
@click.option("--train-dir", default=None, type=str, help="The directory to save checkpoints")
@click.option('--gpus', default="0", type=str, help="Gpus to use")
@click.option('--register-file', default=None, type=str, help="Register_file")
@click.option('--save-every', default=None, type=int, help="Number of epochs to save once")
def train(cfg_file, train_dir, gpus, load, seed, register_file, save_every):
    # Set the device
    gpu_list = [int(gpu) for gpu in gpus.split(",")]
    if not gpu_list:
        device = torch.device("cpu")
        LOGGER.info("Current Device: CPU")
    else:
        _set_gpu(gpu_list[0])
        device = torch.device("cuda:{}".format(gpu_list[0]) if torch.cuda.is_available() else "cpu")
    # Register from the file
    if register_file is not None:
        register(register_file)
    # Set the seed
    if seed is not None:
        LOGGER.info("Setting random seed: %d.", seed)
        _set_seed(seed)
    # Load the configuration
    config = load_yaml(cfg_file)
    config = prepare(config, DEVICE, train-dir, gpus, save_every)
    # Make the train dir
    if train_dir is not None:
        write_yaml(os.path.join(train_dir, "train_config.yaml"), config)
        log_file = os.path.join(train_dir, "train.log")
        _logger.addFile(log_file)
    
    # Instantiate the trainer
    Trainer = get_trainer(config["trainer_type"], **config)
    if load is not None:
        Trainer.load(load)
    Trainer.train()
    return Trainer

main.add_command(train)

@click.command(help="Test Model")
@click.argument("cfg_file", required=True, type=str)
@click.option("--seed", default=None, type=int, help="The random seed to run testing")
@click.option("--load", required=True, type=str, help="The directory to load checkpoint")
@click.option("--test-dir", default=None, type=str, help="The directory to save log and configuration")
@click.option('--device', default="cuda", type=click.Choice(['cpu', 'cuda']), help="cpu or cuda")
@click.option('--gpus', default="0", type=str, help="Gpus to use")
@click.option('--dataset', default="test", type=str, help="Datasets to test on")
@click.option('--register-file', default=None, type=str, help="Register_file")
def test(cfg_file, test_dir, load, device, gpus, dataset, seed, register_file=None):
    # Set the device
    gpu_list = [int(gpu) for gpu in gpus.split(",")]
    if not gpu_list:
        device = torch.device("cpu")
        LOGGER.info("Current Device: CPU")
    else:
        _set_gpu(gpu_list[0])
        device = torch.device("cuda:{}".format(gpu_list[0]) if torch.cuda.is_available() else "cpu")  
    # Register from the file
    if register_file is not None:
        register(register_file)
    # Set the seed
    if seed is not None:
        LOGGER.info("Setting random seed: %d.", seed)
        _set_seed(seed)
    # Load the configuration
    config = load_yaml(cfg_file)
    config = prepare(config, device, testdir, gpus, save_every=None)
    # Make the test dir
    if test_dir is not None:
        write_yaml(os.path.join(test_dir, "test_config.yaml"), config)
        log_file = os.path.join(test_dir, "test.log")
        _logger.addFile(log_file)
    
    # Instantiate the trainer
    Trainer = get_trainer(config["trainer_type"], **config)
    Trainer.load(load)
    Trainer.test(dataset)
    return Trainer

main.add_command(test)


@click.command(help="Show All The Registered Components")
@click.option('--register_file', default=None, type=str, help="Register_file")
def components(register_file):
    if register_file is not None:
        register(register_file)
    print("DLtorch Components")
    print('-- Adv Attackers:', get_attacker_attrs() if len(get_attacker_attrs()) != 0 else None)
    print('-- Datasets:', get_dataset_attrs() if len(get_dataset_attrs()) != 0 else None)
    print('-- Lr_Scedulers:', get_scheduler_attrs() if len(get_scheduler_attrs()) != 0 else None)
    print('-- Objectives', get_objective_attrs() if len(get_objective_attrs()) != 0 else None)
    print('-- Optimizers:', get_optimizer_attrs() if len(get_optimizer_attrs()) != 0 else None)
    print('-- Trainers:', get_trainer_attrs()) if len(get_trainer_attrs()) != 0 else None
    print('-- Models:', get_model_attrs() if len(get_model_attrs()) != 0 else None)

main.add_command(components)

if __name__ == '__main__':
    main()