import yaml

def load_yaml(dir):
    with open(dir, 'r') as f:
        file = yaml.load(f, Loader=yaml.FullLoader)
    return file