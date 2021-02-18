import yaml
from typing import Union


def load_yaml(path: str) -> Union[dict, bool]:
    r"""Load YAML file into dictionary.

    Args:
        path (str): Path to load YAML file from
    Return:
        yml (dict): Dictionary representation of YAML file, False if failure
    """
    try:
        with open(path, 'r') as f:
            yml = yaml.load(f, Loader=yaml.FullLoader)
        return yml
    except:
        return False


def save_yaml(path: str, dic: dict) -> None:
    r"""Save dictionary object into YAML file.

    Args:
        path (str): Path to write YAML file to, recommended postfix '.yml' or '.yaml'
        dic (dict): Dictionary to be written to YAML file
    """
    with open(path, 'w') as f:
        yaml.dump(dic, f)
