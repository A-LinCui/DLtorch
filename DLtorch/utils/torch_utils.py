import torch

def accuracy(outputs, labels):
    assert len(outputs) == len(labels), "Dimensions of outputs and labels must correspond. Outputs size: {} Labels size:{}".format(outputs.shape, labels.shape)
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    return correct / len(outputs)

def correct(outputs, labels):
    assert len(outputs) == len(labels), "Dimensions of outputs and labels must correspond. Outputs size: {} Labels size:{}".format(outputs.shape, labels.shape)
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    return correct

def primary_test(model, dataloader, criterion):
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
    if only_trainable:
        return sum(p.numel() for p in model.parameters())
    else:
        sum(p.numel() for p in model.parameters() if p.requires_grad)