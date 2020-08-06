import os

from DLtorch.component import *
from DLtorch.utils.python_utils import load_yaml, write_yaml

def prepare(config, device, dir, gpus):
    if device is not None:
        assert device in ["cuda", "cpu"], "Device should be 'cuda' or 'cpu'."
        config["device"] = device
    if dir is not None:
        config["path"] = dir
    if gpus is not None:
        config["gpus"] = gpus
    if not os.path.exists(dir):
        os.mkdir(dir)
    return config

def train(config, traindir=None, device=None, gpus=None, checkpoint_path=None):
    config = load_yaml(config)
    config = prepare(config, device, traindir, gpus)
    write_yaml(os.path.join(traindir, "train_config.yaml"), config)
    Trainer = get_trainer(config["trainer_type"], **config)
    if checkpoint_path is not None:
        Trainer.load(checkpoint_path)
    Trainer.train()

def test(config, testdir=None, checkpoint_path=None, device=None, gpus=None, dataset=["train", "test"]):
    assert checkpoint_path is not None, "No available checkpoint."
    config = load_yaml(config)
    config = prepare(config, device, testdir, gpus)
    write_yaml(os.path.join(testdir, "test_config.yaml"), config)
    Trainer = get_trainer(config["trainer_type"], **config)
    Trainer.load(checkpoint_path)
    Trainer.test(dataset)