from typing import List

import numpy as np
import pandas as pd


def dataframe_duv_running_average(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    r"""Computes running average of duv vectors.

    Args:
        df (pd.DataFrame): Dataframe of the form {"folder": str, "file": str, "vid": int, "frame": int, "duv": list (4x2)}
        window (int, optional): Window size for running average. Defaults to 20.

    Returns:
        pd.DataFrame: Dataframe with running average of duv, duv split into individual columns.
    """

    def duv_label() -> List[str]:
        return [f"duv_{i}_{j}" for i in range(4) for j in range(2)]

    df_running_averge = df
    df_running_averge["duv"] = df_running_averge["duv"].apply(
        lambda x: np.array(x).flatten()
    )

    df_running_averge = pd.DataFrame(
        df_running_averge["duv"].to_list(), columns=duv_label()
    )
    df_running_averge = pd.concat(
        [df[["folder", "file", "vid", "frame"]], df_running_averge], axis=1
    )

    df_running_averge[duv_label()] = (
        df_running_averge[["vid"] + duv_label()]
        .groupby("vid")
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(drop=True)
    )

    return df_running_averge
