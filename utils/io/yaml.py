import yaml
import os


def load_yaml(path):
    try:
        with open(path, 'r') as stream:
            yml = yaml.load(stream, Loader=yaml.FullLoader)
        return yml
    except:
        return False


def save_yaml(path, dic):
    with open(os.path.join(path), "w") as f:
        yaml.dump(dic, f)
