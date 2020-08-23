import yaml
import os
from contextlib import contextmanager
from collections import OrderedDict

import matplotlib.pyplot as plt

@contextmanager
def nullcontext():
    yield

def load_yaml(path):
    with open(path, 'r') as f:
        file = yaml.load(f, Loader=yaml.FullLoader)
    return file

def write_yaml(path, config):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

def list_average(list, total):
    return [list[i] / total for i in range(len(list))]

def list_merge(list_1, list_2):
    assert len(list_1) == len(list_2), "The length of two lists is different."
    return [list_1[i] + list_2[i] for i in range(len(list_1))]

def do_nothing():
    pass

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def is_empty(self):
        return self.cnt == 0

class EnsembleAverageMeters(object):

    def __init__(self):
        self.AverageMeters = None

    def is_empty(self):
        return self.AverageMeters is None

    def update(self, perfs, n=1):
        if self.is_empty():
            self.AverageMeters = OrderedDict([(name, AvgrageMeter()) for name in perfs])
        [self.AverageMeters[name].update(val, n) for name, val in perfs.items()]

    def avgs(self):
        return OrderedDict((name, val.avg) for name, val in self.AverageMeters.items()) if not self.is_empty() else None

    def items(self):
        return self.AverageMeters.items() if not self.is_empty() else None

    def reset(self):
        self.AverageMeters = None

class recorder(object):
    def __init__(self, list_names=None, perfs_names=None):
        assert list_names is not None or perfs_names is not None, "Recorder must get names for initialization."
        self.update_epochs = []
        self.recorder = []
        self.list_names = list_names
        self.perfs_names = perfs_names

    def update(self, epoch, statistic):
        assert isinstance(statistic, list), "Second input of update function should be a list."
        self.recorder.append(statistic)
        self.update_epochs.append(epoch)

    def is_empty(self):
        return len(self.update_epochs) == 0

    def get_value(self, name):
        if self.is_empty():
            return None
        elif name in self.list_names:
            idx = self.list_names.index(name)
            return [self.recorder[i][idx] for i in range(len(self.recorder))]
        elif name in self.perfs_names:
            return [self.recorder[i][-1][name] for i in range(len(self.recorder))]

    def get_names(self):
        if self.list_names is None:
            return self.perfs_names
        elif self.perfs_names is None:
            return self.list_names
        else:
            return self.list_names + self.perfs_names

class train_recorder(object):
    def __init__(self, types, list_names, perfs_names=None):
        self.types = types
        self.list_names = list_names
        self.perfs_names = perfs_names
        self.recorders = {}
        for _type in types:
            self.recorders[_type] = recorder(list_names, perfs_names)

    def update(self, _type, epoch, statistic):
        assert _type in self.types, "No type {} in recorder.".format(_type)
        assert isinstance(statistic, list), "Second input of update function should be a list."
        self.recorders[_type].update(epoch, statistic)

    def get_value(self, _type, name):
        assert _type in self.types, "No type {} in recorder.".format(_type)
        return self.recorders[_type].get_value(name)

    def get_names(self):
        if self.list_names is None:
            return self.perfs_names
        elif self.perfs_names is None:
            return self.list_names
        else:
            return self.list_names + self.perfs_names

    def add_type(self, _type):
        self.recorders[_type] = recorder(self.list_names, self.perfs_names)
        self.types.append(_type)

    def draw_curves(self, path=None, show=False):
        item_num = len(self.list_names) + len(self.perfs_names)
        line_num = len(self.recorders.keys())
        plt.figure()
        row = 1
        for item in self.get_names():
            line = 1
            for _type in self.recorders.keys():
                plt.subplot(line_num, item_num, row + item_num * (line - 1))
                try:
                    plt.plot(self.recorders[_type].update_epochs, self.recorders[_type].get_value(item), color="red")
                except:
                    do_nothing()
                plt.xlabel("epoch")
                plt.ylabel("{}-{}".format(_type, item))
                line += 1
            row += 1
        if path is not None:
            plt.savefig(os.path.join(path, "curves.png"))
        if show:
            plt.show()