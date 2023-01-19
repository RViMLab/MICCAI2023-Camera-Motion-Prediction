import yaml


def load_yaml(path: str) -> dict:
    r"""Load YAML file into dictionary.

    Args:
        path (str): Path to load YAML file from
    Return:
        yml (dict): Dictionary representation of YAML file
    """
    try:
        with open(path, "r") as f:
            print(f"Reading file from {path}")
            yml = yaml.load(f, Loader=yaml.FullLoader)
        return yml
    except:
        raise RuntimeError("Failed to load {}".format(path))


def save_yaml(path: str, dic: dict) -> None:
    r"""Save dictionary object into YAML file.

    Args:
        path (str): Path to write YAML file to, recommended postfix '.yml' or '.yaml'
        dic (dict): Dictionary to be written to YAML file
    """
    with open(path, "w") as f:
        print(f"Writing file to {path}")
        yaml.dump(dic, f)
