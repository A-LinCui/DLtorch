import os
import argparse

from torch.nn import DataParallel

import DLtorch.utils as utils
import DLtorch.component as component
import DLtorch.datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="resnet_train_config.yaml", help="config file")
    parser.add_argument('--gpus', type=str, default="0", help="gpus")
    args = parser.parse_args()

    # Set the logger
    log = utils.logger(name="train", save_path=None, whether_stream=True, whether_file=False)
    # Load the config file
    config = utils.load_yaml(args.config)
    # Set used gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    # Load the data
    dataset = component.get_dataset("Cifar10")
    dataloader = dataset.dataloader(**config["Dataset"]["Dataloader"])

    log.info("Load data {} from {}.".format(config["Dataset"]["type"], DLtorch.config.config().datasets[config["Dataset"]["type"]]))
    # Load the models
    model = component.get_model("LeNet").cuda()
    if len(str(args.gpus)) > 1:
        model = DataParallel(model)
    params = list(model.parameters())
    optimizer = component.get_optimizer("SGD", params=params, lr=float(config["Train"]["lr"]), weight_decay=float(config["Train"]["weight_decay"]),
                          momentum=float(config["Train"]["momentum"]), nesterov=True)
    scheduler = component.get_scheduler("MultiStepLR", optimizer=optimizer,
                            milestones=[int(config["Train"]["epochs"] / 2), int(config["Train"]["epochs"] * 3 / 4),
                                        int(config["Train"]["epochs"] * 7 / 8)],
                            gamma=float(config["Train"]["scheduler"]["gamma"]))
    objective = component.objective.get_objective("ClassificationObjective")
    total, correct, train_loss = 0, 0, 0
    for (images, labels) in dataloader["train"]:
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        loss = objective.get_loss(images, outputs, labels, model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Count acc,loss on trainset
        correct += utils.accuracy(outputs, labels) * len(images)
        total += labels.shape[0]
        train_loss += loss.item()