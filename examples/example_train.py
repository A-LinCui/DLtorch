from DLtorch.train.final_trainer import *
from DLtorch.utils.python_utils import load_yaml

if __name__ == "__main__":
    config = load_yaml("train_config.yaml")
    Trainer = FinalTrainer(**config)
    Trainer.train()