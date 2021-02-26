import pickle


def load_pickle(path: str) -> dict:
    r"""Loads a binary file into a dictionary.

    Args:
        path (str): Path to load binary file from
    Return:
        dic (dict): Dictionary object
    """
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except:
       raise RuntimeError('Failed to load {}'.format(path))


def save_pickle(path: str, dic: dict) -> None:
    r"""Saves a dictinoary into a binary file.

    Args:
        path (str): Path to write dict to, recommended postfix '.pkl'
        dic (dict): Dictionary object to be written to binary file
    """
    with open(path, 'wb') as f:
        pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)
