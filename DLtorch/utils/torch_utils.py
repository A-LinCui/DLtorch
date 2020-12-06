# DLtorch Framework
# Author: Junbo Zhao <zhaojb17@mails.tsinghua.edu.cn>.

import torch
import torch.nn as nn
from torchviz import make_dot

class CrossEntropyLabelSmooth(nn.Module):
    """CrossEntropy with Label Smoothing."""
    
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
    # Test a model's accuracy on the dataset basically.
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

def get_params(model, only_trainable=False):
    # Get the parameter number of the model.
    # If only_trainable is true, only trainable parameters will be counted.
    if not only_trainable:
        return sum(p.numel() for p in model.parameters())
    else:
        sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy(outputs, targets, topk=(1,)):
    # Get top-k accuracy on the data batch.
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

def random_tensor(shape):
    # Randomly generate a tensor with the given shape
    return torch.randn(shape)

def plot_arch(net, shape, device):
    # Plot the architecture of the given model. Input shape supported by the model should be given as "shape".
    # For example, to plot a typical cifar10 model, the given shape should be (1, 3, 32, 32).
    # Current device of the net should be given.
    x = random_tensor(shape).to(device)
    vis_graph = make_dot(net(x), params=dict(net.named_parameters()))
    vis_graph.view()
