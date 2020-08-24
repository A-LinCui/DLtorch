# !bai/usr/bin/python

import torch

from DLtorch.train.base import BaseFinalTrainer
from DLtorch.utils.python_utils import *
from DLtorch.utils.torch_utils import accuracy

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
        
    # ---- API ----
    def train(self):

        self.log.info("DLtorch Train : FinalTrainer  Start training···")
        # Init the all of the components.
        self.init_component()
        # Count the parameters of the mode to be trained.
        self.count_param()
        # Add statistics recorder
        self.recorder = train_recorder(types=["train", "test"], list_names=["loss", "top-1-acc", "top-5-acc", "reward"],
                                       perfs_names=self.objective.perf_names)

        # If using early stop, add validation part to training statistics.
        if self.early_stop:
            self.log.info("Using early stopping.")
            self.recorder.add_type("valid")

        for epoch in range(self.last_epoch + 1, self.epochs + 1):

            # Print the current learning rate.
            self.log.info("epoch: {} learning rate: {}".format(epoch, self.optimizer_kwargs["lr"] if not hasattr(self, "scheduler") else self.scheduler.get_lr()[0]))

            # Train on training set for one epoch.
            loss, accs, perfs, reward = self.train_epoch(self.dataloader["train"])
            self.recorder.update("train", epoch, [loss, accs["top-1"], accs["top-5"], reward, perfs])

            # Step the learning rate if scheduler isn't none.
            if hasattr(self, "scheduler"):
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

    def test(self, dataset):
        self.log.info("DLtorch Trainer : FinalTrainer  Start testing···")
        self.count_param()
        self.init_component()
        assert hasattr(self, "model") and hasattr(self, "optimizer"), \
            "At least one component in 'model, optimizer' isn't available. Please load or initialize them before testing."
        assert "valid" not in dataset or self.early_stop, \
            "No validation dataset available or early_stop hasn't set to be true. Check the configuration."
        for data_type in dataset:
            loss, accs, perfs, reward = self.infer(self.dataloader[data_type], data_type)

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