import torch

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
    if not only_trainable:
        return sum(p.numel() for p in model.parameters())
    else:
        sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy(outputs, targets, topk=(1,)):
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