# !bai/usr/bin/python
import os

import matplotlib.pyplot as plt
import torch

from DLtorch.train.base import BaseFinalTrainer
import DLtorch.utils as utils
from DLtorch.utils.python_utils import *

class CNNFinalTrainer(BaseFinalTrainer):
    NAME = "CNNFinalTrainer"

    def __init__(self, device, gpus,
                 epochs, grad_clip, eval_no_grad, early_stop,
                 model, model_kwargs,
                 dataset, dataset_kwargs, dataloader_kwargs,
                 objective, objective_kwargs,
                 optimizer_type, optimizer_kwargs,
                 scheduler, scheduler_kwargs,
                 save_as_state_dict, path,
                 test_every, valid_every, save_every, report_every, trainer_type="CNNFinalTrainer"
                 ):
        super(CNNFinalTrainer, self).__init__(device, gpus, model, model_kwargs, dataset, dataset_kwargs, dataloader_kwargs,
                                           objective, objective_kwargs, optimizer_type, optimizer_kwargs,
                                           scheduler, scheduler_kwargs, path, trainer_type)

        # Set other training configs
        self.epochs = epochs
        self.test_every = test_every
        self.valid_every = valid_every
        self.save_every = save_every
        self.report_every = report_every
        # Other configs
        self.save_as_state_dict = save_as_state_dict
        self.early_stop = early_stop
        self.grad_clip = grad_clip
        self.eval_no_grad = eval_no_grad

        self.last_epoch = 0
        name = ["train", "test", "valid"] if self.early_stop else ["train", "test"]
        self.training_statistics = {item: {"epoch": [], "loss": [], "acc": [], "reward": [], "perf": []} for item in name}
        
    # ---- API ----
    def train(self):
        self.log.info("DLtorch Train : FinalTrainer  Start training···")
        self.init_component()
        self.count_param()

        if self.early_stop:
            self.log.info("Using early stopping.")
            best_reward, best_epoch, best_loss, best_acc, best_perf = 0, 0, 0, 0, 0
            """
            If we load the checkpoint before training and there is no validation statistics in the checkpoint,
            there will be no key called "valid" in self.training_statistics. Therefore, we should add it.
            """
            if "valid" not in list(self.training_statistics.keys()):
                self.training_statistics["valid"] = {"epoch": [], "acc": [], "loss": [], "reward": [], "perf": []}

        for epoch in range(self.last_epoch, self.epochs):

            # Print the current learning rate.
            if not hasattr(self, "scheduler"):
                self.log.info("epoch: {} learning rate: {}".format(epoch, self.component_kwargs["optimizer"]["lr"]))
            else:
                self.log.info("epoch: {} learning rate: {}".format(epoch, self.scheduler.get_lr()[0]))

            # Train on training set for one epoch.
            self.update_statistics("train", epoch + 1, self.train_epoch(self.dataloader["train"], epoch))

            # Step the learning rate if scheduler isn't none.
            if hasattr(self, "scheduler"):
                self.scheduler.step()

            # Test on validation set and save the model with the best performance.
            if (epoch + 1) % self.valid_every == 0 and self.early_stop:
                loss, accuracy, perf, reward = self.infer(self.dataloader["valid"], epoch, "valid")
                self.update_statistics("valid", epoch + 1, (loss, accuracy, perf, reward))
                if reward > best_reward or best_reward == 0:
                    best_reward, best_loss, best_acc, best_perf, best_epoch = reward, loss, accuracy, perf, epoch
                    if self.path is not None:
                        save_path = os.path.join(self.path, "best")
                        self.save(save_path)
                self.log.info("best_valid_epoch: {} acc:{:.5f} loss:{:.5f} reward:{:.5f} perf: {}".
                              format(best_epoch + 1, best_acc, best_loss, best_reward,
                                     ";".join(["{}: {:.3f}".format(n, v) for n, v in zip(self.objective.perf_names, best_perf)])))

            # Test on test dataset
            if (epoch + 1) % self.test_every == 0:
                self.update_statistics("test", epoch + 1, self.infer(self.dataloader["test"], epoch))

            # Save the current model.
            if (epoch + 1) % self.save_every == 0 and self.path is not None:
                save_path = os.path.join(self.path, str(epoch))
                self.save(save_path)

            self.last_epoch += 1

    def test(self, dataset):
        self.log.info("DLtorch Trainer : FinalTrainer  Start testing···")
        self.count_param()
        self.init_component()
        assert hasattr(self, "model") and hasattr(self, "optimizer"), \
            "At least one component in 'model, optimizer' isn't available. Please load or initialize them before testing."
        assert "valid" not in dataset or self.early_stop, \
            "No validation dataset available or early_stop hasn't set to be true. Check the configuration."
        for data_type in dataset:
            loss, accuracy, perf, reward = self.infer(self.dataloader[data_type], 0, data_type)

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
        # Save the training statistics
        torch.save(self.training_statistics, os.path.join(path, "statistics.pt"))
        self.draw_curves(path, show=True)
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
            self.training_statistics = torch.load(statistics_path)
            self.log.info("Load training statistics from {}".format(statistics_path))

    def update_statistics(self, name, epoch, statistic):
        # Save the statistics after training, testing or validating for an epoch
        self.training_statistics[name]["epoch"].append(epoch)
        for key, consequence in zip(list(self.training_statistics[name].keys())[1:], statistic):
            self.training_statistics[name][key].append(consequence if key != "perf" else [consequence])

    def draw_curves(self, path=None, show=False):
        """
        Draw curves for all the statistics.
        """
        plt.figure()
        item_num = len(self.training_statistics["train"].keys()) + len(self.objective.perf_names) - 2
        line_num = len(self.training_statistics.keys())

        row = 1
        for item in self.training_statistics["train"].keys():
            if item not in ["epoch", "perf"]:
                line = 1
                for dataset in self.training_statistics.keys():
                    plt.subplot(line_num, item_num, row + item_num * (line -1))
                    plt.plot(self.training_statistics[dataset]["epoch"], self.training_statistics[dataset][item], color="red")
                    plt.xlabel("epoch")
                    plt.ylabel("{}-{}".format(dataset, item))
                    line += 1
                row += 1

        for item in range(len(self.objective.perf_names)):
            line = 1
            for dataset in self.training_statistics.keys():
                plt.subplot(line_num, item_num, row + item_num * (line-1))
                try:
                    perf = [self.training_statistics[dataset]["perf"][num][item] for num in range(len(self.training_statistics[dataset]["epoch"]))]
                    plt.plot(self.training_statistics[dataset]["epoch"], perf, color="red")
                    plt.xlabel("epoch")
                    plt.ylabel("{}-perf-{}".format(dataset, self.objective.perf_names[item]))
                    line += 1
                except:
                    line += 1

        if path is not None:
            plt.savefig(os.path.join(path, "curves.png"))
        if show:
            plt.show()

    # ---- Inner Functions ----
    def train_epoch(self, data_queue=None, epoch=0):
        self.model.train()
        start_train = False
        report_batch = [int(i * self.report_every * len(data_queue)) for i in range(1, int(1 / self.report_every))]
        batch_num = 0
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
            if batch_num in report_batch:
                self.log.info("train_epoch: {} process: {} / {} acc: {:.5f} loss:{:.5f} reward:{:.5f} perf: {}".
                              format(epoch + 1, batch_num, len(data_queue), correct / total, loss / total, reward / total,
                                     ";".join(["{}: {:.3f}".format(n, v) for n, v in zip(self.objective.perf_names,
                                                                                     list_average(perf, total))])))
        loss, accuracy, reward, perf = float(loss / total), float(correct / total), float(reward / total), list_average(perf, total)
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
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
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
        report_batch = [int(i * self.report_every * len(data_queue)) for i in range(1, int(1 / self.report_every))]
        batch_num = 0
        context = torch.no_grad() if self.eval_no_grad else nullcontext()

        with context:
            for (inputs, targets) in data_queue:
                if not start_infer:
                    loss, correct, total, perf, reward = self.infer_batch(inputs, targets)
                    start_infer = True
                else:
                    batch_loss, batch_correct, batch_size, batch_perf, batch_reward = self.infer_batch(inputs, targets)
                    loss += batch_loss
                    correct += batch_correct
                    total += batch_size
                    perf = list_merge(perf, batch_perf)
                    reward += batch_reward
                batch_num += 1
                if batch_num in report_batch:
                    self.log.info("{}_epoch: {} process: {} / {} acc: {:.5f} loss:{:.5f} reward:{:.5f} perf: {}".format(
                        _type, epoch + 1, batch_num, len(data_queue), correct / total, loss / total, reward / total, ";".
                            join(["{}: {:.3f}".format(n, v) for n, v in zip(self.objective.perf_names, list_average(perf, total))])))

        loss, accuracy, reward, perf = float(loss / total), float(correct / total), float(reward / total), list_average(
            perf, total)
        self.log.info("{}_epoch: {} process: {} / {} acc: {:.5f} loss:{:.5f} reward:{:.5f} perf:{}".format(
            _type, epoch + 1, batch_num, len(data_queue), accuracy, loss, reward, ";".join(
                ["{}: {:.3f}".format(n, v) for n, v in zip(self.objective.perf_names, perf)])))

        return loss, accuracy, perf, reward