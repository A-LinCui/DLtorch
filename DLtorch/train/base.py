# !bai/usr/bin/python
import os
import abc

import torch

import DLtorch.component as component
from DLtorch.utils.logger import logger
from DLtorch.utils.torch_utils import get_params

class BaseFinalTrainer(object):
    NAME = "BaseFinalTrainer"

    def __init__(self, device, gpus, model, model_kwargs,
                 dataset, dataset_kwargs, dataloader_kwargs,
                 objective, objective_kwargs,
                 optimizer_type, optimizer_kwargs,
                 scheduler, scheduler_kwargs, path, trainer_type
                 ):

        # Makedir
        self.path = path
        if path is not None and not os.path.exists(self.path):
            os.mkdir(self.path)

        # Set the log
        self.log = logger(name=trainer_type, save_path=os.path.join(self.path, "trainer.log"),
                          whether_stream=True, whether_file=True) if path is not None else \
            logger(name="Final Training", whether_stream=True)
        self.log.info("DLtorch Framework: Constructing {} ···".format(trainer_type))

        # Set all the components
        self.model_type = model
        self.dataset_type = dataset
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler
        self.objective_type = objective
        self.model_kwargs = model_kwargs
        self.objective_kwargs = objective_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.dataset_kwargs = dataset_kwargs
        self.dataloader_kwargs = dataloader_kwargs

        # Set the device
        self.gpus = gpus
        self.device = device
        self.set_gpus()

    # ---- virtual APIs to be implemented in subclasses ----
    @abc.abstractmethod
    def train(self):
        """
        Do the actual training task of your trainer.
        """

    @abc.abstractmethod
    def test(self, dataset):
        """
        Test the newest model on different datasets.
        """

    @abc.abstractmethod
    def save(self, path):
        """
        Save the trainer state to disk.
        """

    @abc.abstractmethod
    def load(self, path):
        """
        Load the trainer state from disk.
        """

    @abc.abstractmethod
    def infer(self, data_queue, _type):
        """
        Infer the model.
        """

    # ---- Construction Helper ---
    def set_gpus(self):
        if self.device == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpus)
            self.log.info("GPU information: {}".format(self.gpus))
        else:
            self.log.info("Using CPU.")

    def count_param(self, only_trainable=False):
        """
        Count the parameter number for the model.
        """
        self.param = get_params(self.model, only_trainable=only_trainable)
        self.log.info("Parameter number for current model: {}M".format(self.param / 1.0e6))

    def init_model(self):
        # Initialize the model
        assert self.model_type is not None, "Available model not found. Check the configuration."
        self.log.info("Initialize Model: {}".format(self.model_type))
        self.model = component.get_model(self.model_type, **self.model_kwargs).to(self.device) \
            if self.model_kwargs is not None else component.get_model(self.model_type).to(self.device)
        if len(str(self.gpus)) > 1:
            self.model = torch.nn.DataParallel(self.model)

    def init_optimizer(self):
        # Initialize the optimizer.
        assert self.optimizer_type is not None, "Available optimizer not found. Check the configuration."
        self.log.info("Initialize Optimizer: {}".format(self.optimizer_type))
        self.optimizer = component.get_optimizer(self.optimizer_type, params=list(self.model.parameters()),
                                                 **self.optimizer_kwargs) if self.optimizer_kwargs is not None \
            else component.get_optimizer(self.optimizer_type, params=list(self.model.parameters()))

    def init_scheduler(self):
        # Initialize the scheduler.
        if self.scheduler_type is not None:
            self.log.info("Initialize Scheduler: {}".format(self.scheduler_type))
            self.scheduler = component.get_scheduler(self.scheduler_type, **self.scheduler_kwargs,
                                                     optimizer=self.optimizer) if self.scheduler_kwargs is not None \
                else component.get_scheduler(self.scheduler_type, optimizer=self.optimizer)

    def init_objective(self):
        # Initialize the objective.
        assert self.objective_type is not None, "Available objective not found. Check the configuration."
        self.log.info("Initialize Objective: {}".format(self.objective_type))
        self.objective = component.objective.get_objective(self.objective_type, **self.objective_kwargs) \
            if self.objective_kwargs is not None else component.get_objective(self.objective_type)

    def init_dataset(self):
        # Initialize the dataset.
        assert self.dataset_type is not None, "Available dataset not found. Check the configuration."
        self.log.info("Initialize Dataset: {}".format(self.dataset_type))
        self.dataset = component.get_dataset(self.dataset_type, **self.dataset_kwargs) \
            if self.dataset_kwargs is not None else component.get_dataset(self.dataset_type)
        assert self.dataloader_kwargs is not None, "Available dataloader config not found. Check the configuration."
        self.log.info("Initialize Dataloader.")
        self.dataloader = self.dataset.dataloader(**self.dataloader_kwargs)

    def init_component(self):
        """
         Init all the components that haven't been initialized, including model, dataset, dataloader, optimizer, scheduler and objective.
         Note that schedule is optional.
        """
        init_functions = {"model": self.init_model,
                          "dataset": self.init_dataset,
                          "optimizer": self.init_optimizer,
                          "scheduler": self.init_scheduler,
                          "objective": self.init_objective
                          }

        for component in init_functions.keys():
            if not hasattr(self, component):
                init_functions[component]()