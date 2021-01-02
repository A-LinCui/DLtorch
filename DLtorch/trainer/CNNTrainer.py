# -*- coding:utf-8 -*-

import os
import collections

import torch
import torch.utils.data as data

import DLtorch
from DLtorch.trainer.base import BaseTrainer
from DLtorch.utils.common_utils import AvgrageMeter, EnsembleAverageMeters, nullcontext
from DLtorch.utils.torch_utils import accuracy


class CNNTrainer(BaseTrainer):
    def __init__(
        self,
        # Components
        model,
        dataset,
        dataloader_kwargs: dict,
        objective,
        optimizer_type: str,
        optimizer_kwargs: dict = {},
        lr_scheduler_type: str = None,
        lr_scheduler_kwargs: dict = {},
        # Training cfgs
        path: str = None,
        device: str = "cuda",
        gpu_list: list = [0],
        epochs: int = 100,
        save_every: int = 10,
        save_as_state_dict: bool = True,
        report_every: int = 50,
        test_every: int = 1,
        grad_clip: float = 5.0,
        eval_no_grad: bool = True,
        early_stop: bool = False,
        trainset_portion: list = [0.8, 0.2]
        ):
        super(CNNTrainer, self).__init__(
            model, 
            dataset, dataloader_kwargs,
            objective,
            optimizer_type, optimizer_kwargs,
            lr_scheduler_type, lr_scheduler_kwargs,
            path, 
            device, gpu_list,
            epochs,
            save_every, save_as_state_dict,
            report_every, test_every,
            grad_clip,
            eval_no_grad
            )
        self.early_stop = early_stop
        self.portion = trainset_portion

        # Split the datasets and construct dataloaders
        if self.early_stop:
            assert isinstance(self.portion, list), "Early stop is used. 'Trainset_portion'[list] is required for dividing the original training set."
            assert sum(self.portion) == 1.0, "Early stop is used. 'Trainset_portion' invalid. The sum of it should be 1.0."
            self.logger.info("Early stop is used. Split trainset into [train/valid] = {}".format(self.portion))
            self.dataset.datasets["train"], self.dataset.datasets["valid"] = torch.utils.data.random_split(
                self.dataset.datasets["train"], [int(len(self.dataset.datasets["train"]) * _pt) for _pt in self.portion])
            self.best_reward, self.best_acc, self.best_loss, self.best_epoch, self.best_perfs = 0, 0, 0, 0, None

        self.dataloader = {}
        for _dataset in self.dataset.datasets.keys():
            self.dataloader[_dataset] = data.DataLoader(self.dataset.datasets[_dataset], **dataloader_kwargs)
        
        self.last_epoch = 0

    
    # ---- API ----
    def train(self):
        self.logger.info("Start training···")

        for epoch in range(self.last_epoch + 1, self.epochs + 1):
            self.logger.info("Epoch: {}; Learning rate: {}".format(epoch, 
                self.optimizer_kwargs["lr"] if self.lr_scheduler is None else self.lr_scheduler.get_lr()[0])
                )

            # Train on training set for one epoch.
            loss, accs, perfs, reward = self.train_epoch(self.dataloader["train"])

            # Step the learning rate if scheduler isn't none.
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Test on validation set and save the model with the best performance.
            if self.early_stop:
                loss, accs, perfs, reward = self.infer(self.dataloader["valid"], "valid")

                if reward > self.best_reward:
                    self.best_reward, self.best_acc, self.best_loss, self.best_epoch, self.best_perfs = reward, accs, loss, epoch, perfs
                    if self.path is not None:
                        save_path = os.path.join(self.path, "best")
                        self.save(save_path)

                self.logger.info("best_valid_epoch: {}; top-1: {:.5f}; loss: {:.5f}; reward:{:.5f}; perfs: {}".\
                    format(self.best_epoch, self.best_acc["top-1"], self.best_loss, self.best_reward, ";".join(["{}: {:.3f}".format(n, v) for n, v in self.best_perfs.items()])))

            # Test on test dataset
            if epoch % self.test_every == 0:
                loss, accs, perfs, reward = self.infer(self.dataloader["test"])
            
            # Save the current model
            if epoch % self.save_every == 0 and self.path is not None:
                save_path = os.path.join(self.path, str(epoch))
                self.save(save_path)

            self.last_epoch += 1

        if self.path is not None:
            save_path = os.path.join(self.path, "final")
            self.save(save_path)


    def test(self, split: str):
        assert split in self.dataloader.keys(), "No subdataset: {} in the dataset.".format(split)
        self.logger.info("Start testing···")
        loss, accs, perfs, reward = self.infer(self.dataloader[split], split)


    def save(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save the model
        if self.save_as_state_dict:
            model_path = os.path.join(path, "model_state.pt")
            torch.save(self.model.state_dict(), model_path)
        else:
            model_path = os.path.join(path, "model.pt")
            torch.save(self.model, model_path)

        # Save the optimizer
        torch.save({"epoch": self.last_epoch, "optimizer": self.optimizer.state_dict()}, os.path.join(path, "optimizer.pt"))
        
        # Save the lr scheduler
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(path, "lr_scheduler.pt"))

        # Save valid information
        if self.early_stop:
            torch.save(
                {
                    "best_reward": self.best_reward, 
                    "best_acc": self.best_acc, 
                    "best_loss": self.best_loss, 
                    "best_epoch": self.best_epoch, 
                    "best_perfs": self.best_perfs
                    },
                os.path.join(path, "valid_info.pt")
                )

        self.logger.info("Save the checkpoint at {}".format(os.path.abspath(path)))


    def load(self, path: str):
        assert os.path.exists(path), "The loading path '{}' doesn't exist.".format(path)
        
        # Load the model
        if os.path.isdir(path):
            if os.path.exists(os.path.join(path, "model.pt")):
                model_path = os.path.join(path, "model.pt")
                self.model = torch.load(model_path, map_location=torch.device("cpu"))
            else:
                model_path = os.path.join(path, "model_state.pt")
                model_state = torch.load(model_path, map_location=torch.device("cpu"))
                self.model.load_state_dict(model_state)
        elif path.endswith("model.pt"):
            model_path = path
            self.model = torch.load(model_path, map_location=torch.device("cpu"))
        elif path.endswith("model_state.pt"):
            model_path = path
            model_state = torch.load(model_path, map_location=torch.device("cpu"))
            self.model.load_state_dict(model_state)
        
        self.model = self.model.to(self.device)
        if self.device != torch.device("cpu") and len(self.gpu_list) > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.logger.info("Load model from {}".format(os.path.abspath(model_path)))

        # Load the optimizer
        optimizer_path = os.path.join(path, "optimizer.pt")
        if os.path.exists(optimizer_path) and os.path.isdir(path):
            optimizer_checkpoint = torch.load(optimizer_path, map_location=torch.device("cpu"))
            self.optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
            self.last_epoch = optimizer_checkpoint["epoch"]
            self.logger.info("Load optimizer from {}".format(os.path.abspath(optimizer_path)))

        # Load the lr scheduler
        lr_scheduler_path = os.path.join(path, "lr_scheduler.pt")
        if os.path.exists(lr_scheduler_path) and os.path.isdir(path):
            self.lr_scheduler.load_state_dict(torch.load(lr_scheduler_path, map_location=torch.device("cpu")))
            self.logger.info("Load lr scheduler from {}".format(lr_scheduler_path))
        
        # Load valid information
        valid_info_path = os.path.join(path, "valid_info.pt")
        if os.path.exists(valid_info_path) and os.path.isdir(path) and self.early_stop:
            valid_info_checkpoint = torch.load(valid_info_path, map_location=torch.device("cpu"))
            self.best_reward = valid_info_checkpoint["best_reward"]
            self.best_acc = valid_info_checkpoint["best_acc"]
            self.best_loss = valid_info_checkpoint["best_loss"]
            self.best_epoch = valid_info_checkpoint["best_epoch"]
            self.best_perfs = valid_info_checkpoint["best_perfs"]
            self.logger.info("Load valid information from {}".format(os.path.abspath(valid_info_path)))
        

    # ---- Inner Functions ----
    def train_epoch(self, data_queue):
        self.model.train()
        loss, reward = AvgrageMeter(), AvgrageMeter()
        accs, perfs = EnsembleAverageMeters(), EnsembleAverageMeters()

        for i, (inputs, targets) in enumerate(data_queue):
            batch_loss, batch_accs, batch_perfs, batch_reward = self.train_batch(inputs, targets)
            # Update statistics
            batch_size = len(targets)
            loss.update(batch_loss, batch_size)
            reward.update(batch_reward, batch_size)
            accs.update(batch_accs, batch_size)
            perfs.update(batch_perfs, batch_size)

            if (i + 1) % self.report_every == 0 or i == len(data_queue) - 1:
                self.logger.info("train_epoch: {}; process: {} / {}; top-1: {:.5f}; loss:{:.5f}; reward:{:.5f}; perfs: {}".\
                    format(self.last_epoch + 1, i + 1, len(data_queue), accs.avgs()["top-1"], loss.avg, reward.avg, ";".join(["{}: {:.3f}".format(n, v) for n, v in perfs.avgs().items()])))

        return loss.avg, accs.avgs(), perfs.avgs(), reward.avg


    def train_batch(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        accs = collections.OrderedDict({"top-1": prec1, "top-5": prec5})
        loss = self.objective.get_loss(inputs, outputs, targets, self.model)
        perfs_value = self.objective.get_perfs(inputs, outputs, targets, self.model)
        perfs = collections.OrderedDict([(name, perf) for name, perf in zip(self.objective.perf_names, perfs_value)])
        reward = self.objective.get_reward(perfs_value)
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return loss.item(), accs, perfs, reward


    def infer_batch(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        accs = collections.OrderedDict({"top-1": prec1, "top-5": prec5})
        loss = self.objective.get_loss(inputs, outputs, targets, self.model)
        perfs_value = self.objective.get_perfs(inputs, outputs, targets, self.model)
        perfs = collections.OrderedDict([(name, perf) for name, perf in zip(self.objective.perf_names, perfs_value)])
        reward = self.objective.get_reward(perfs_value)
        return loss.item(), accs, perfs, reward


    def infer(self, data_queue, _type="test"):
        self.model.eval()
        loss, reward = AvgrageMeter(), AvgrageMeter()
        accs, perfs = EnsembleAverageMeters(), EnsembleAverageMeters()

        context = torch.no_grad() if self.eval_no_grad else nullcontext()

        with context:
            for i, (inputs, targets) in enumerate(data_queue):
                batch_loss, batch_accs, batch_perfs, batch_reward = self.infer_batch(inputs, targets)
                batch_size = len(targets)
                loss.update(batch_loss, batch_size)
                reward.update(batch_reward, batch_size)
                accs.update(batch_accs, batch_size)
                perfs.update(batch_perfs, batch_size)

                if (i + 1) % self.report_every == 0 or i == len(data_queue) - 1:
                    self.logger.info("{}_epoch: {}; process: {} / {}; top-1: {:.5f}; loss:{:.5f}; reward:{:.5f}; perfs: {}".\
                        format(_type, self.last_epoch + 1, i + 1, len(data_queue), accs.avgs()["top-1"], loss.avg, reward.avg, ";".join(["{}: {:.3f}".format(n, v) for n, v in perfs.avgs().items()])))

        return loss.avg, accs.avgs(), perfs.avgs(), reward.avg
