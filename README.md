## DLtorch: An Extensible Deep Learning Framework based on Pytorch

## Introduction
With the rapid development of deep learning technology in the field of artificial intelligence, 
a series of deep learning frameworks, such as TensorFlow, Jittor, Pytorch, have been born. 
Among which, pytorch as a rising star, has the advantages of python-like syntax and ease of network construction.
However, using the native pytorch framework to train a deep neural network (DNN) still requires a lot of codes and work. 
For this reason, we developed DLtorch, an extensible deep learning framework based on pytorch, to quickly implement DNN algorithms.

### Components of a DNN training system
Generally, a typical DNN training system, can be categorized into components below. 
And in DLtorch framework, the interface between these components is somehow well-defined.
```
Components:
  -- dataset             The dataset to train and test on.
  -- criterion           Crierion used for training.
  -- model               Neural network.
  -- optimizer           Define how to optimize the parameters of a network.
  -- lr_scheduler        Define how to adjust the learning rate while training.
  -- objective           Define the loss, accuracy, reward and other statistics while training.
  -- trainer             Define how to train a model.
```
Flow of DLtorch is shown below.
![Flow](flow.png)

## Install
Using a virtual python environment is encouraged. For example, with Anaconda, you could run `conda create -n DLtorch python==3.7.3 pip` first.
* Supported python versions: 3.6, 3.7
* Supported Pytorch versions: >=1.0.0, <1.5.0
To install `DLtorch`, run `pip install DLtorch` directly or `python setup.py build && python setup.py install` after cloning this project.

## Usage
After installation, you can run `DLtorch --help` to see what sub-commands are available.
Output of an example run (version 1.0.0):
```
Usage: DLtorch [OPTIONS] COMMAND [ARGS]...
  
  The DLtorch framework command line interface.

Options:
  --version                Show the version and exit.
  --local_rank  INTEGER    The rank of this process  [default: -1]
  --help                   Show this message and exit.

Commands:
  components               Show All The Registered Components
  test                     Test Model
  train                    Train Model
```

### Run DNN Training
You can run `DLtorch train --help` to see how to train a model.
Output of an example run (version 1.0.0):
```
Usage: DLtorch [OPTIONS] COMMAND [ARGS]...
  
  Train model

Options:
  --seed INTEGER           The random seed to run training
  --load  TEXT             The directory to load checkpoint (If not given, train from scratch)
  --traindir TEXT          The directory to save checkpoints (If not given, nothing will be saved)
  --device [cpu|cuda]      cpu or cuda [default: cuda]
  --gpus  TEXT             Gpus to use [default: 0]
  --register_file TEXT     Register_file
  --save-every INTEGER     Number of epochs to save once
  --help                   Show this message and exit.
```
Try training a ResNet-18 net on cifar10 from scratch, the results (including configuration backup, training log, checkpoints, statistics, training curves) will be saved in `<TRAIN_DIR>`). Nothing will be saved if `<TRAIN_DIR>` isn't given.

```
DLtorch train examples/cifar10_basic.yaml --gpus 0 --seed 123 --save-every <SAVE_EVERY> --train-dir <TRAIN_DIR>
```

### Run DNN Testing
You can run `DLtorch test --help` to see how to test a model.
Output of an example run (version 1.0.0):
```
Usage: DLtorch [OPTIONS] COMMAND [ARGS]...
  
  Test model

Options:
  --seed INTEGER           The random seed to run testing
  --load  TEXT             The directory to load checkpoint
  --testdir TEXT           The directory to save log and configuration
  --device [cpu|cuda]      cpu or cuda [default: cuda]
  --gpus  TEXT             Gpus to use [default: 0]
  --dataset TEXT           Datasets to test on [default: test]
  --register_file TEXT     Register_file
  --help                   Show this message and exit.
```
Try testing a pretrained resnet-18 net on cifar10, the results (including configuration backup, testing log) will be saved in `<TEST_DIR>`). Nothing will be saved if `<TEST_DIR>` isn't given.

```
DLtorch test examples/cifar10_basic.yaml --gpus 0 --load <CHECKPOINT_DIR> --dataset test --device cuda --testdir <TEST_DIR>
```

### Show DLtorch Registered Components
Only components registered by DLtorch framework can be used while training and testing. Components registered in our framework include those supported by Pytorch and those implemented by us. 
Run `DLtorch components` to see what components are implemented.
```
Usage: DLtorch components [OPTIONS]

  Show All The Registered Components

Options:
  --register_file TEXT  Register_file
  --help                Show this message and exit.
```

### Components Registeration
DLtorch framework itself only support components implemented in our framework and Pytorch. To use new designed components, they have to be registered into DLtorch before using.
In command line, we provide a registeration interface `--register_file`. To use new components, it must be defined in a python file. And a function named `register`, in which components are rigistered into DLtorch, must be provided in it.
We provide APIs for registering different components as below. Details can be seen in our code.

```
DLtorch.components.regist_Criterion  # Regist a criterion
DLtorch.components.regist_scheduler  # Regist a learning rate scheduler
DLtorch.components.regist_model      # Regist a model
DLtorch.components.regist_objective  # Regist an objective
DLtorch.components.regist_optimizer  # Regist an optimizer
DLtorch.components.regist_trainer    # Regist a trainer
```

For examples, try training a ResNet-18 net on cifar10 from scratch using a new designed component `ExampleNewObjective` defined and registed in `examples/example_new_objective.py`.

```
DLtorch train examples/register_example.yaml --gpus 0 --seed 123 --save-every <SAVE_EVERY> --train-dir <TRAIN_DIR> --register_file examples/example_new_objective.py 
```

### Adversarial Attack & Adversarial Training
We implement adversarial attack and adversarial training in our framework. It's split into two parts: adversarial example generator and adversarial objective. Currently, we only support FGSM and PGD attack. You can implement your own by subclassing `BaseAdvGenerator` and `ClassificationAdversarialObjective`. Welcome to fork us and share your implementation.

Try adversarially training a resnet-18 net on cifar10 with PGD method.

```
DLtorch test examples/cifar10_adv.yaml --gpus 0 --load <CHECKPOINT_DIR> --dataset test --device cuda --testdir <TEST_DIR>
```

### Notation
* At present, DLtorch framework only support CNN. Welcome to help improve this framework, whether it's contributing code or proposing improvement plans. 
* Due to developer's level limitation, there may be bugs in the code. If you encounter, please contact to resolve.

### Acknowlegdements
Our implementation refers to the codes in repositories below. Thanks for their help.
* [DARTS](https://github.com/quark0/darts)
* [awnas](https://github.com/walkerning/aw_nas)
