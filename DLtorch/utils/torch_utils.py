# -*- coding:utf-8 -*-

import warnings
from collections import OrderedDict

import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchviz import make_dot
from torchstat import ModelStat, ModelHook, StatTree, StatNode


class CrossEntropyLabelSmooth(nn.Module):
    """
    CrossEntropy with Label Smoothing.
    """
    
    def __init__(self, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        num_classes = int(inputs.shape[-1])
        log_probs = nn.LogSoftmax(dim=1)(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.smooth) * targets + self.smooth / num_classes
        loss = - (targets * log_probs).mean(0).sum()
        return loss


def primary_test(model, dataloader, criterion):
    """
    Test a model's accuracy on the dataset basically.
    """

    model.eval()
    correct, total, loss = 0, 0, 0
    with torch.no_grad():
        for (images, labels) in dataloader:
            batch_size = len(images)
            total += batch_size
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            correct += accuracy(outputs, labels) * batch_size
            loss += criterion(outputs, labels).item()
    return loss / total, correct / total


def accuracy(outputs, targets, topk=(1,)):
    """
    Get top-k accuracy on the data batch.
    """

    maxk = max(topk)
    batch_size = len(targets)
    _, predicts = outputs.topk(maxk, 1, True, True)
    predicts = predicts.t()
    correct = predicts.eq(targets.view(1, -1).expand_as(predicts))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0/batch_size).item())
    return res


# ---- Tools for Model Analysis ----
class _ModelHook(ModelHook):
    def __init__(self, model, input_size, device):
        super(_ModelHook, self).__init__(model, input_size)
        self._model = model
        x = torch.rand(1, *self._input_size).to(device)  # add module duration time
        self._model.eval()
        self._model(x)


class _ModelStat(ModelStat):
    def __init__(self, model, input_size, query_granularity=1):
        super(_ModelStat, self).__init__(model, input_size, query_granularity)

    def _analyze_model(self):
        model_hook = _ModelHook(self._model, self._input_size)
        leaf_modules = model_hook.retrieve_leaf_modules()
        stat_tree = self.convert_leaf_modules_to_stat_tree(leaf_modules)
        collected_nodes = stat_tree.get_collected_stat_nodes(self._query_granularity)
        return collected_nodes

    def convert_leaf_modules_to_stat_tree(self, leaf_modules):
        assert isinstance(leaf_modules, OrderedDict)

        create_index = 1
        root_node = StatNode(name='root', parent=None)
        for leaf_module_name, leaf_module in leaf_modules.items():
            names = leaf_module_name.split('.')
            for i in range(len(names)):
                create_index += 1
                stat_node_name = '.'.join(names[0:i+1])
                parent_node = self.get_parent_node(root_node, stat_node_name)
                node = StatNode(name=stat_node_name, parent=parent_node)
                parent_node.add_child(node)
                if i == len(names) - 1:  # leaf module itself
                    input_shape = leaf_module.input_shape.numpy().tolist()
                    output_shape = leaf_module.output_shape.numpy().tolist()
                    node.input_shape = input_shape
                    node.output_shape = output_shape
                    node.Flops = leaf_module.Flops.numpy()[0]
        return StatTree(root_node)
    
    def get_parent_node(self, root_node, stat_node_name):
        assert isinstance(root_node, StatNode)
        node = root_node
        names = stat_node_name.split('.')
        for i in range(len(names) - 1):
            node_name = '.'.join(names[0:i+1])
            child_index = node.find_child_index(node_name)
            assert child_index != -1
            node = node.children[child_index]
        return node


def get_params(model, only_trainable=False):
    """
    Get the parameter number of the model.
    If only_trainable is true, only trainable parameters will be counted.
    """

    if not only_trainable:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_arch(net, shape, device, path: str, view: bool = False):
    """
    Plot the architecture of the given model. Input shape supported by the model should be given as "shape".
    For example, to plot a typical cifar10 model, the given shape should be (x, 3, 32, 32).
    Current device of the net should be given.
    """

    x = Variable(torch.randn(shape, requires_grad=True)).to(device)
    vis_graph = make_dot(net(x), params=dict(net.named_parameters()))
    try:
        vis_graph.render(filename = net.__class__.__name__, directory = path, view=view)
        return True
    except:
        warnings.warn("Fail to render with graphviz. Maybe it's because your operation system is Windows", category=None, stacklevel=1, source=None)
        return False


def get_flops(model, shape):
    """
    Get FLOPs of the model.
    Input shape should be the shape of a single input, instead of the shape of a batch.
    For example, for a typical model trained on CIFAR-10, it should be (3, 32, 32)
    """

    model_analyzer = _ModelStat(model, shape) 
    collected_nodes = model_analyzer._analyze_model()
    flops = 0
    for node in collected_nodes:
        flops += node.Flops
    return flops