import yaml

def load_yaml(dir):
    with open(dir, 'r') as f:
        file = yaml.load(f, Loader=yaml.FullLoader)
    return file

def list_average(list, total):
    return [list[i] / total for i in range(len(list))]

def list_merge(list_1, list_2):
    assert len(list_1) == len(list_2), "The length of two lists is different."
    return [list_1[i] + list_2[i] for i in range(len(list_1))]