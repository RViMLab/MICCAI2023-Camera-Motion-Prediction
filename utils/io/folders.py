import pathlib
import os
import glob
import re

import pandas as pd


def generate_path(prefix):
    if not os.path.exists(prefix):
        pathlib.Path(prefix).mkdir(parents=True, exist_ok=True)

def scan2df(folder, postfix='.jpg'):
    # scan folder for images and return dataframe
    df = pd.DataFrame(columns={'folder', 'files'})

    for file in glob.glob(os.path.join(folder, '*{}'.format(postfix))):
        dic = {
            'folder': folder,
            'files': os.path.basename(file)
        }

        df = df.append(dic, ignore_index=True)

    return df

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    r"""Sorts in human order, see
        https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    
    Example:
        sorted_list = sorted(list, key=natural_keys)
    """
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
