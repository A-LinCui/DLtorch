# !bai/usr/bin/python
import os

import torch

import DLtorch.utils as utils
import DLtorch.component as component
from DLtorch.utils.logger import logger
from DLtorch.utils.python_utils import *

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
                 test_every=1, valid_every=1, save_every=1, report_every=0.2
                 ):
        # Makedir
        self.path = path
        if path is not None and not os.path.exists(self.path) :
            os.mkdir(self.path)
        # Set the log
        self.log = logger(name="Final Training", save_path=os.path.join(self.path, "train.log"),
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
        self.test_every = test_every
        self.valid_every = valid_every
        self.save_every = save_every
        self.report_every = report_every
        # Initialize components
        self.set_gpus()
        self.init_component()
        # Other configs
        self.save_as_state_dict = save_as_state_dict
        self.early_stop = early_stop
        self.grad_clip = grad_clip
        self.eval_no_grad = eval_no_grad

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        if self.save_as_state_dict:
            path = os.path.join(path, "model_state.pt")
            torch.save(self.model.state_dict(), path)
        else:
            path = os.path.join(path, "model.pt")
            torch.save(self.model, path)
        self.log.info("Save the model as {}".format(os.path.abspath(path)))

    def load(self, path):
        assert os.path.exists(path), "The loading path '{}' doesn't exist.".format(path)
        model_path = os.path.join(path, "model.pt") if os.path.isdir(path) else path
        if not os.path.exists(model_path):
            model_path = os.path.join(path, "model_state.pt")
            self.model.load_state_dict(model_path)
        else:
            self.model = torch.load(model_path, map_location=torch.device("cpu"))
        self.log.info("Load model from {}".format(os.path.abspath(model_path)))
        self.model.to(self.device)
        if len(str(self.gpus)) > 1:
            self.model = torch.nn.DataParallel(self.model)

    def train(self):
        self.log.info("DLtorch Train : FinalTrainer  Start training···")
        if self.early_stop:
            self.log.info("Using early stopping.")
        for epoch in range(self.epochs):
            if self.early_stop:
                best_reward, best_epoch, best_loss, best_acc, best_perf = 0, 0, 0, 0, 0
            loss, accuracy, perf, reward = self.train_epoch(self.dataloader["train"], epoch)
            if self.scheduler_type is not None:
                self.scheduler.step()

            if (epoch + 1) % self.save_every == 0 and self.path is not None:
                save_path = os.path.join(self.path, str(epoch))
                self.save(save_path)

            if (epoch + 1) % self.valid_every == 0 and self.early_stop:
                loss, accuracy, perf, reward = self.infer(self.dataloader["valid"], epoch, "valid")
                if reward > best_reward or best_reward == 0:
                    best_reward, best_loss, best_acc, best_perf, best_epoch = reward, loss, accuracy, perf, epoch
                    if self.path is not None:
                        save_path = os.path.join(self.path, "best")
                        self.save(save_path)
                self.log.info("best_valid_epoch: {} acc:{:.5f} loss:{:.5f} reward:{:.5f} perf: {}".
                              format(best_epoch + 1, best_acc, best_loss, best_reward,
                                     ";".join(["{}: {:.3f}".format(n, v) for n, v in zip(self.objective.perf_names, best_perf)])))

            if (epoch + 1) % self.test_every == 0:
                loss, accuracy, perf, reward = self.infer(self.dataloader["test"], epoch, "test")

    def train_epoch(self, data_queue=None, epoch=0):
        self.model.train()
        start_train = False
        batch_num, report_batch = 0, 0
        for (inputs, targets) in data_queue:
            if not start_train:
                loss, correct, total, perf, reward = self.train_batch(inputs, targets)
                start_train = True
            else:
                batch_loss, batch_correct, batch_size, batch_perf, batch_reward = self.train_batch(inputs, targets)
                loss += batch_loss
                correct += batch_correct
                total += batch_size
                perf = list_merge(perf, batch_perf)
                reward += batch_reward
            batch_num += 1
            report_batch += 1
            if report_batch > self.report_every * len(data_queue):
                report_batch = 0
                self.log.info("train_epoch: {} process: {} / {} acc: {:.5f} loss:{:.5f} reward:{:.5f} perf: {}".
                              format(epoch + 1, batch_num, len(data_queue), correct / total, loss / total, reward / total,
                                     ";".join(["{}: {:.3f}".format(n, v) for n, v in zip(self.objective.perf_names,
                                                                                     list_average(perf, total))])))
        loss /= total
        accuracy = correct / total
        perf = list_average(perf, total)
        reward /= total
        self.log.info("train_epoch: {} process: {} / {} acc: {:.5f} loss:{:.5f} reward:{:.5f} perf: {}".
                      format(epoch + 1, batch_num, len(data_queue), accuracy, loss, reward,
                             ";".join(["{}: {:.3f}".format(n, v) for n, v in zip(self.objective.perf_names, perf)])))

        return loss, accuracy, perf, reward

    def train_batch(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        loss = self.objective.get_loss(inputs, outputs, targets, self.model)
        perf = self.objective.get_perfs(inputs, outputs, targets, self.model)
        reward = self.objective.get_reward(perf)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), utils.correct(outputs, targets), len(inputs), perf, reward

    def infer_batch(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        loss = self.objective.get_loss(inputs, outputs, targets, self.model)
        perf = self.objective.get_perfs(inputs, outputs, targets, self.model)
        reward = self.objective.get_reward(perf)
        return loss.item(), utils.correct(outputs, targets), len(inputs), perf, reward

    def infer(self, data_queue=None, epoch=0, _type="test"):
        self.model.eval()
        start_infer = False
        batch_num, report_batch = 0, 0
        for (inputs, targets) in data_queue:
            if not start_infer:
                loss, correct, total, perf, reward = self.infer_batch(inputs, targets)
                start_infer = True
            else:
                batch_loss, batch_correct, batch_size, batch_perf, batch_reward = self.train_batch(inputs, targets)
                loss += batch_loss
                correct += batch_correct
                total += batch_size
                perf = list_merge(perf, batch_perf)
                reward += batch_reward
            batch_num += 1
            report_batch += 1
            if report_batch > self.report_every * len(data_queue):
                report_batch = 0
                self.log.info(
                    "{}_epoch: {} process: {} / {} acc: {:.5f} loss:{:.5f} reward:{:.5f} perf: {}".
                    format(_type, epoch + 1, batch_num, len(data_queue), correct / total, loss / total, reward / total,
                           ";".join(["{}: {:.3f}".format(n, v) for n, v in zip(self.objective.perf_names,
                                                                               list_average(perf, total))])))
        loss /= total
        accuracy = correct / total
        perf = list_average(perf, total)
        reward /= total
        self.log.info("{}_epoch: {} process: {} / {} acc: {:.5f} loss:{:.5f} reward:{:.5f} perf:{}".
            format(_type, epoch + 1, batch_num, len(data_queue), accuracy, loss, reward,
                   ";".join(["{}: {:.3f}".format(n, v) for n, v in zip(self.objective.perf_names, perf)])))

        return loss, accuracy, perf, reward

    # ---- Construction Helper ----
    def set_gpus(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpus)

    def init_component(self):
        """
        Init all the components, including model, dataset, dataloader, optimizer, scheduler and objective.
        Note that schedule is optional.
        """

        self.log.info("GPU information: {}".format(self.gpus))

        assert self.model_type is not None, "Available model not found. Check the configuration."
        self.log.info("Initialize Model: {}".format(self.model_type))
        if self.model_kwargs is not None:
            self.model = component.get_model(self.model_type, **self.model_kwargs).to(self.device)
        else:
            self.model = component.get_model_cls(self.model_type)().to(self.device)
        if len(str(self.gpus)) > 1:
            self.model = torch.nn.DataParallel(self.model)

        assert self.dataset_type is not None, "Available dataset not found. Check the configuration."
        self.log.info("Initialize Dataset: {}".format(self.dataset_type))
        if self.dataset_kwargs is not None:
            self.dataset = component.get_dataset(self.dataset_type, **self.dataset_kwargs)
        else:
            component.get_objective_cls(self.dataset_type)()

        assert self.dataset_kwargs is not None, "Available dataloader config not found. Check the configuration."
        self.log.info("Initialize Dataloader.")
        self.dataloader = self.dataset.dataloader(**self.dataloader_kwargs)

        assert self.optimizer_type is not None, "Available optimizer not found. Check the configuration."
        self.log.info("Initialize Optimizer: {}".format(self.optimizer_type))
        if self.optimizer_kwargs is not None:
            self.optimizer = component.get_optimizer(self.optimizer_type, params=list(self.model.parameters()), **self.optimizer_kwargs)
        else:
            self.optimizer = component.get_optimizer(self.optimizer_type, params=list(self.model.parameters()))

        # Initialize the scheduler
        if self.scheduler_type is not None:
            self.log.info("Initialize Scheduler: {}".format(self.scheduler_type))
            if self.scheduler_kwargs is not None:
                self.scheduler = component.get_scheduler(self.scheduler_type, optimizer=self.optimizer, **self.scheduler_kwargs)
            else:
                self.scheduler = component.get_scheduler(self.scheduler_type, optimizer=self.optimizer)

        # Initialize the objective
        assert self.objective_type is not None, "Available objective not found. Check the configuration."
        self.log.info("Initialize Objective: {}".format(self.objective_type))
        if self.objective_kwargs is not None:
            self.objective = component.objective.get_objective(self.objective_type, **self.objective_kwargs)
        else:
            self.objective = component.get_objective_cls(self.objective_type)()

if __name__ == "__main__":
    from DLtorch.utils.python_utils import load_yaml
    config = load_yaml("train_config.yaml")
    Trainer = FinalTrainer(**config)
    Trainer.train()