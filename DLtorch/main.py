from __future__ import print_function

import os
import functools
import click
import importlib
import sys

from DLtorch.component import *
from DLtorch.utils import set_seed, load_yaml, write_yaml
from DLtorch.version import __version__

def register(path):
    # Automatically run the "register" function defined in the "path(.py)" file to register new components into DLtorch.
    p, f = os.path.split(os.path.abspath(path))
    sys.path.append(p)
    module = importlib.import_module(f[:-3])
    module.register()
    sys.path.remove(p)

def register_components(path):
    if isinstance(path, str):
        register(path)
    else:
        [register(one_path) for one_path in path]

def prepare(config, device, dir, gpus, seed):
    if device is not None:
        assert device in ["cuda", "cpu"], "Device should be 'cuda' or 'cpu'."
        config["device"] = device
    if dir is not None:
        config["path"] = dir
    else:
        config["path"] = None
    if gpus is not None:
        config["gpus"] = gpus
    if dir is not None and not os.path.exists(dir):
        os.mkdir(dir)
    return config

click.option = functools.partial(click.option, show_default=True)
@click.group(help="The awnas NAS framework command line interface. "
             "Use `AWNAS_LOG_LEVEL` environment variable to modify the log level.")
@click.version_option(version=__version__)
@click.option("--local_rank", default=-1, type=int,
              help="the rank of this process")
def main(local_rank):
    pass

@click.command(help="Train Model")
@click.argument("cfg_file", required=True, type=str)
@click.option("--seed", default=None, type=int,
              help="The random seed to run training")
@click.option("--load", default=None, type=str,
              help="the directory to load checkpoint")
@click.option("--traindir", default=None, type=str,
              help="the directory to save checkpoints")
@click.option('--device', default="cuda", type=str,
              help="cpu or cuda")
@click.option('--gpus', default="0", type=str, help="gpus")
@click.option('--register_file', default=None, type=str or list, help="register_file(s)")

def train(cfg_file, traindir, device, gpus, load, seed, register_file):
    set_seed(seed)
    if register_file is not None:
        register_components(register_file)
    config = load_yaml(cfg_file)
    config = prepare(config, device, traindir, gpus, seed)
    if traindir is not None:
        write_yaml(os.path.join(traindir, "train_config.yaml"), config)
    Trainer = get_trainer(config["trainer_type"], **config)
    if load is not None:
        Trainer.load(load)
    Trainer.train()
    return Trainer
main.add_command(train)

@click.command(help="Test Model")
@click.argument("cfg_file", required=True, type=str)
@click.option("--seed", default=None, type=int,
              help="The random seed to run training")
@click.option("--load", default=None, type=str,
              help="the directory to load checkpoint")
@click.option("--testdir", default=None, type=str,
              help="the directory to save checkpoints")
@click.option('--device', default="cuda", type=str,
              help="cpu or cuda")
@click.option('--gpus', default="0", type=str, help="gpus")
@click.option('--dataset', default=["train", "test"], type=list, help="datasets to test on")
@click.option('--register_file', default=None, type=str or list, help="register_file(s)")
def test(cfg_file, testdir, load, device, gpus, dataset, seed, register_file):
    assert load is not None, "No available checkpoint."
    set_seed(seed)
    if register_file is not None:
        register_components(register_file)
    config = load_yaml(cfg_file)
    config = prepare(config, device, testdir, gpus, seed)
    if testdir is not None:
        write_yaml(os.path.join(testdir, "test_config.yaml"), config)
    Trainer = get_trainer(config["trainer_type"], **config)
    Trainer.load(load)
    Trainer.test(dataset)
    return Trainer
main.add_command(test)

if __name__ == '__main__':
    main()