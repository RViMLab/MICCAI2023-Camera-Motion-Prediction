import glob
import os
import pathlib
import re
from typing import List, Union

import pandas as pd


def generate_path(prefix: str) -> None:
    if not os.path.exists(prefix):
        print(f"Generating path at {prefix}")
        pathlib.Path(prefix).mkdir(parents=True, exist_ok=True)


def scan2df(folder: str, postfix: str = ".jpg") -> pd.DataFrame:
    # scan folder for images and return dataframe
    df = pd.DataFrame(columns={"folder", "file"})

    print(f"Scanning {folder} for {postfix} files")
    for file in glob.glob(os.path.join(folder, "*{}".format(postfix))):
        df = pd.concat(
            [df, pd.DataFrame({"folder": [folder], "file": [os.path.basename(file)]})],
            ignore_index=True,
        )

    return df


def recursive_scan2df(folder: str, postfix: str = ".jpg") -> pd.DataFrame:
    # scan folder for images and return dataframe
    df = pd.DataFrame(columns={"folder", "file"})

    print(f"Scanning {folder} recursively for {postfix} files")
    for root, subdirs, files in os.walk(folder):
        files = [x for x in files if postfix in x]
        if files:
            dic_list = [
                {"folder": root.replace(folder, "").strip("/"), "file": x}
                for x in files
            ]
            df = df.append(dic_list, ignore_index=True)
    return df


def atoi(text: str) -> Union[int, str]:
    return int(text) if text.isdigit() else text


def natural_keys(text: str) -> List[int]:
    r"""Sorts in human order, see
        https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside

    Example:
        sorted_list = sorted(list, key=natural_keys)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]
