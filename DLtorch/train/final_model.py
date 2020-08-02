import os

import torch

import DLtorch.utils as utils
import DLtorch.component as component
from DLtorch.utils.logger import logger

class FinalTrainer(object):
    NAME = "FinalTrainer"

    def __init__(self, device, gpus,
                 epochs=200, grad_clip=5.0, eval_no_grad=False, early_stop=False,
                 model=None, model_kwargs=None,
                 dataset=None, dataset_kwargs=None, dataloader_kwargs=None,
                 objective=None, objective_kwargs=None,
                 optimizer_type=None, optimizer_kwargs=None,
                 scheduler=None, scheduler_kwargs=None,
                 save_as_state_dict=False, path=None,
                 ):
        # Set the log
        self.log = logger(name="Final Training", save_path=os.path.join(path, "train.log"),
                          whether_stream=True, whether_file=True) if path is not None else \
            logger(name="Final Training", whether_stream=True)
        self.log.info("DLtorch Framework: Constructing FinalTrainer ···")
        # Set the device
        self.device = device
        self.gpus = gpus
        # Set all the components
        self.model_type = model
        self.model_kwargs = model_kwargs
        self.dataset_type = dataset
        self.dataset_kwargs = dataset_kwargs
        self.dataloader_kwargs = dataloader_kwargs
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_type = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        self.objective_type = objective
        self.objective_kwargs = objective_kwargs
        # Set other training configs
        self.epochs = epochs
        # Initialize components
        self.set_gpus()
        self.init_component()
        # Other configs
        self.save_as_state_dict = save_as_state_dict
        self.early_stop = early_stop
        self.grad_clip = grad_clip
        self.eval_no_grad = eval_no_grad

    def init_component(self):
        self.model = component.get_model(self.model_type, **self.model_kwargs).to(self.device)
        if len(str(self.gpus)) > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.dataset = component.get_dataset(self.dataset_type, **self.dataset_kwargs)
        self.dataloader = self.dataset.dataloader(**self.dataloader_kwargs)
        self.optimizer = component.get_optimizer(self.optimizer_type, params=list(self.model.parameters()),
                                                 **self.optimizer_kwargs)
        self.scheduler = component.get_scheduler(self.scheduler_type, optimizer=self.optimizer, **self.scheduler_kwargs)
        self.objective = component.objective.get_objective("ClassificationObjective")

    def save(self, path):
        if self.save_as_state_dict:
            path = os.path.join(path, "model_state.pt")
            torch.save(self.model.state_dict(), path)
        else:
            path = os.path.join(path, "model.pt")
            torch.save(self.model, path)

    def load(self, path):
        model_path = os.path.join(path, "model.pt") if os.path.isdir(path) else path
        if not os.path.exists(model_path):
            model_path = os.path.join(path, "model_state.pt")
            self.model.load_state_dict(model_path)
        else:
            self.model = torch.load(model_path, map_location=torch.device("cpu"))
        self.log.info("Load model from {}".format(os.path.abspath(model_path)))

    def set_gpus(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpus)

    def train(self, data_queue=None):
        self.model.train()
        data_queue = data_queue if data_queue is None else self.dataloader["train"]

        total, correct, loss = 0, 0, 0

        for epoch in range(self.epochs):
            for (inputs, labels) in data_queue:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = self.model(inputs)
                loss = self.objective.get_loss(inputs, outputs, labels, self.model)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # Count acc,loss on trainset
                correct += utils.accuracy(outputs, labels) * len(inputs)
                total += labels.shape[0]
                loss += loss.item()

    def infer(self, data_queue=None):
        self.model.eval()
        data_queue = data_queue if data_queue is None else self.dataloader["test"]
        total, correct, loss = 0, 0, 0
        for (inputs, labels) in data_queue:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = self.model(inputs)
            loss = self.objective.get_loss(inputs, outputs, labels, self.model)
            correct += utils.accuracy(outputs, labels) * len(inputs)
            total += labels.shape[0]
            loss += loss.item()
        return float(correct / total), float(loss / total)