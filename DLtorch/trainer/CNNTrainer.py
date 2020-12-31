# -*- coding:utf-8 -*-

import torch
import torch.utils.data as data

import DLtorch
from DLtorch.trainer.base import BaseTrainer
from DLtorch.utils.common_utils import *
from DLtorch.utils.python_utils import *
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
            epochs, 
            save_every, save_as_state_dict,
            report_every,
            test_every,
            grad_clip,
            eval_no_grad
            )
        
        self.early_stop = early_stop
        self.portion = trainset_portion

        # Split the datasets and construct dataloaders
        if self.early_stop:
            assert isinstance(self.portion, list), "'Trainset_portion'[list] is required for dividing the original training set if using early stop."
            assert sum(self.portion) == 1.0, "'Trainset_portion' invalid. The sum of it should be 1.0."
            self.logger.info("Using early stop. Split trainset into [train/valid] = {}".format(self.portion))
            self.dataset["train"], self.dataset["valid"] = torch.utils.data.random_split(
                self.dataset["train"], [int(len(self.dataset["train"]) * _pt) for _pt in self.portion)])
        dataloader = {}
        for _dataset in self.dataset.keys():
            dataloader[_dataset] = data.DataLoader(self.dataset[_dataset], **dataloader_kwargs)
        
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
                self.scheduler.step()

            # Test on validation set and save the model with the best performance.
            if self.early_stop and epoch % self.valid_every == 0:
                loss, accs, perfs, reward = self.infer(self.dataloader["valid"], "valid")
                self.recorder.update("valid", epoch, [loss, accs["top-1"], accs["top-5"], reward, perfs])
                if not hasattr(self, "best_reward") or reward > self.best_reward or self.best_reward == 0:
                    self.best_reward, self.best_loss, self.best_acc, self.best_perf, self.best_epoch = \
                        reward, loss, accs, perfs, epoch

                    if self.path is not None:
                        save_path = os.path.join(self.path, "best")
                        self.save(save_path)
                self.log.info("best_valid_epoch: {} top-1: {:.5f} top-5: {:.5f} loss: {:.5f} reward:{:.5f} perf: {}".
                              format(self.best_epoch, self.best_acc["top-1"], self.best_acc["top-5"], self.best_loss,
                                     self.best_reward, ";".join(["{}: {:.3f}".format(n, v) for n, v in self.best_perf.items()])))

            # Test on test dataset
            if epoch % self.test_every == 0:
                loss, accs, perfs, reward = self.infer(self.dataloader["test"])
                self.recorder.update("test", epoch, [loss, accs["top-1"], accs["top-5"], reward, perfs])

            # Save the current model.
            if epoch % self.save_every == 0 and self.path is not None:
                save_path = os.path.join(self.path, str(epoch))
                self.save(save_path)

            self.last_epoch += 1

        if self.path is not None:
            save_path = os.path.join(self.path, "final")
            self.save(save_path)

    def test(self, dataset):
        self.log.info("DLtorch Trainer : FinalTrainer  Start testing···")
        self.count_param()
        self.init_component()
        assert hasattr(self, "model") and hasattr(self, "optimizer"), \
            "At least one component in 'model, optimizer' isn't available. Please load or initialize them before testing."
        assert "valid" not in dataset or self.early_stop, \
            "No validation dataset available or early_stop hasn't set to be true. Check the configuration."
        if isinstance(dataset, list):
            for data_type in dataset:
                loss, accs, perfs, reward = self.infer(self.dataloader[data_type], data_type)
        elif isinstance(dataset, str):
            loss, accs, perfs, reward = self.infer(self.dataloader[dataset], dataset)

    def save(self, path):
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
        # Save the scheduler
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), os.path.join(path, "scheduler.pt"))
        # Save the statistics
        torch.save(self.recorder, os.path.join(path, "statistics.pt"))
        self.recorder.draw_curves(path, show=False)
        self.log.info("Save the checkpoint at {}".format(os.path.abspath(path)))

    def load(self, path):
        assert os.path.exists(path), "The loading path '{}' doesn't exist.".format(path)
        # Load the model
        model_path = os.path.join(path, "model.pt") if os.path.isdir(path) else path
        if os.path.exists(model_path):
            self.model = torch.load(model_path, map_location=torch.device("cpu"))
        else:
            model_path = os.path.join(path, "model_state.pt")
            self.init_model()
            self.model.load_state_dict(model_path)
        self.model.to(self.device)
        self.log.info("Load model from {}".format(os.path.abspath(model_path)))
        # Load the optimizer
        self.init_optimizer()
        optimizer_path = os.path.join(path, "optimizer.pt") if os.path.isdir(path) else None
        if optimizer_path and os.path.exists(optimizer_path):
            optimizer_checkpoint = torch.load(optimizer_path, map_location=torch.device("cpu"))
            self.optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
            self.last_epoch = optimizer_checkpoint["epoch"]
            self.log.info("Load optimizer from {}".format(os.path.abspath(optimizer_path)))
        # Load the scheduler
        self.init_scheduler()
        scheduler_path = os.path.join(path, "scheduler.pt") if os.path.isdir(path) else None
        if scheduler_path and os.path.exists(scheduler_path):
            self.scheduler.load_state_dict(torch.load(scheduler_path, map_location=torch.device("cpu")))
            self.log.info("Load scheduler from {}".format(scheduler_path))
        # Load the statistics
        statistics_path = os.path.join(path, "statistics.pt") if os.path.isdir(path) else None
        if statistics_path and os.path.exists(statistics_path):
            self.recorder = torch.load(statistics_path)
            self.log.info("Load statistic recorder from {}".format(statistics_path))

    # ---- Inner Functions ----
    def train_epoch(self, data_queue):
        self.model.train()
        data_queue_length, batch_num = len(data_queue), 0
        loss, reward = AvgrageMeter(), AvgrageMeter()
        accs, perfs = EnsembleAverageMeters(), EnsembleAverageMeters()
        report_batch = [int(i * self.report_every * data_queue_length) for i in range(1, int(1 / self.report_every))]

        for (inputs, targets) in data_queue:

            batch_size = len(targets)
            batch_num += 1

            batch_loss, batch_accs, batch_perfs, batch_reward = self.train_batch(inputs, targets)
            loss.update(batch_loss, batch_size)
            reward.update(batch_reward, batch_size)
            accs.update(batch_accs, batch_size)
            perfs.update(batch_perfs, batch_size)

            if batch_num in report_batch or batch_num == data_queue_length:
                self.log.info("train_epoch: {} process: {} / {} top-1: {:.5f} top-5: {:.5f} loss:{:.5f} "
                              "reward:{:.5f} perf: {}".format(self.last_epoch + 1, batch_num, len(data_queue), accs.avgs()["top-1"],
                accs.avgs()["top-5"], loss.avg, reward.avg, ";".join(["{}: {:.3f}".format(n, v) for n, v in perfs.avgs().items()])))

        return loss.avg, accs.avgs(), perfs.avgs(), reward.avg

    def train_batch(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        accs = OrderedDict({"top-1": prec1, "top-5": prec5})
        loss = self.objective.get_loss(inputs, outputs, targets, self.model)
        perfs_value = self.objective.get_perfs(inputs, outputs, targets, self.model)
        perfs = OrderedDict([(name, perf) for name, perf in zip(self.objective.perf_names, perfs_value)])
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
        accs = OrderedDict({"top-1": prec1, "top-5": prec5})
        loss = self.objective.get_loss(inputs, outputs, targets, self.model)
        perfs_value = self.objective.get_perfs(inputs, outputs, targets, self.model)
        perfs = OrderedDict([(name, perf) for name, perf in zip(self.objective.perf_names, perfs_value)])
        reward = self.objective.get_reward(perfs_value)
        return loss.item(), accs, perfs, reward

    def infer(self, data_queue, _type="test"):

        self.model.eval()
        data_queue_length, total, batch_num = len(data_queue), 0, 0
        loss, reward = AvgrageMeter(), AvgrageMeter()
        accs, perfs = EnsembleAverageMeters(), EnsembleAverageMeters()
        report_batch = [int(i * self.report_every * data_queue_length) for i in range(1, int(1 / self.report_every))]
        context = torch.no_grad() if self.eval_no_grad else nullcontext()

        with context:
            for (inputs, targets) in data_queue:
                batch_size = len(targets)
                batch_num += 1
                batch_loss, batch_accs, batch_perfs, batch_reward = self.infer_batch(inputs, targets)
                loss.update(batch_loss, batch_size)
                reward.update(batch_reward, batch_size)
                accs.update(batch_accs, batch_size)
                perfs.update(batch_perfs, batch_size)

                if batch_num in report_batch or batch_num == data_queue_length:
                    self.log.info("{}_epoch: {} process: {} / {} top-1: {:.5f} top-5: {:.5f} loss:{:.5f} reward:{:.5f} "
                                  "perf: {}".format(_type, self.last_epoch + 1, batch_num, len(data_queue),
                                                   accs.avgs()["top-1"], accs.avgs()["top-5"], loss.avg, reward.avg,
                                           ";".join(["{}: {:.3f}".format(n, v) for n, v in perfs.avgs().items()])))

        return loss.avg, accs.avgs(), perfs.avgs(), reward.avg
