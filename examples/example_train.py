from DLtorch.main import *

train(config="train_config.yaml", traindir="try", device="cuda", gpus="0")
# test(config="train_config.yaml", checkpoint_path="try/0", testdir="TEST", device="cuda", gpus="0")