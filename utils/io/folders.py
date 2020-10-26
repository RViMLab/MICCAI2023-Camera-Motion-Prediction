import pathlib
import os
import glob

import pandas as pd


def generate_path(prefix):
    if not os.path.exists(prefix):
        pathlib.Path(prefix).mkdir(parents=True, exist_ok=True)

def scan2df(folder, postfix='.jpg'):
    # scan folder for images and return dataframe
    df = pd.DataFrame(columns={'folder', 'file'})

    for file in glob.glob(os.path.join(folder, '*{}'.format(postfix))):
        dic = {
            'folder': folder,
            'file': os.path.basename(file)
        }

        df = df.append(dic, ignore_index=True)

    return df
