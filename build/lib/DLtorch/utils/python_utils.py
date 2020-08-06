import yaml
from contextlib import contextmanager

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