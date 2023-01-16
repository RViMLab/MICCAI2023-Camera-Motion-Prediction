import os
import random
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from decord import VideoLoader, bridge, cpu
from decord._ffi.ndarray import DECORDContext
from torch.utils.data import Dataset


class DecordDataset(Dataset):
    def __init__(
        self,
        data_df: pd.DataFrame,
        prefix: str,
        seeds: bool = False,
        transforms: Callable = None,
        ctx: DECORDContext = cpu(0),
        shape: Tuple[int] = (1, 240, 320, 3),  # BxHxWxC
        interval: int = 0,
        skip: int = 0,
        shuffle: int = 0,
    ):
        bridge.set_bridge("torch")
        self._data_df = data_df.sort_values(["folder", "file"]).reset_index(drop=True)
        self._transforms = transforms
        if self._transforms:
            raise NotImplemented("DecordDataset: No transforms implemented.")
        self._seeds = seeds

        # Create video loader
        self._paths = [
            os.path.join(prefix, row.folder, row.file)
            for _, row in self._data_df.iterrows()
        ]

        self._video_loader = VideoLoader(
            uris=self._paths,
            ctx=ctx,
            shape=shape,
            interval=interval,
            skip=skip,
            shuffle=shuffle,
        )

    def __getitem__(self, idx):
        if self._seeds:
            seed = random.randint(0, np.iinfo(np.int32).max)  # set random seeed
        else:
            seed = idx

        frames, idcs = next(self._video_loader)

        # # TODO: add transforms

        return frames, idcs

    def __len__(self):
        return len(self._video_loader)


if __name__ == "__main__":
    import endoscopy
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from utils.io import recursive_scan2df

    prefix = "/media/martin/Samsung_T5/data/endoscopic_data/cholec80/sample_videos"
    data_df = recursive_scan2df(prefix, ".mp4")
    shape = [100, 480, 640, 3]

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    detector = endoscopy.BoundingCircleDetector(device=device)

    ds = DecordDataset(data_df=data_df, prefix=prefix, shape=shape)
    dl = DataLoader(ds, batch_size=1, num_workers=0)  # num_workers must be 0
    log_df = pd.DataFrame(columns=["vid", "frame", "center", "radius"])

    for batch in tqdm(dl):
        frames, idcs = batch
        frames, idcs = frames[0], idcs[0]

        frames = frames.to(device).float().permute(0, 3, 1, 2) / 255.0
        center, radius = detector(frames, reduction="max")
        # box = endoscopy.max_rectangle_in_circle(imgs.shape, center, radius)
        # crops = kornia.geometry.crop_and_resize(imgs, box, [240, 320])

        seq_data = {
            "vid": idcs[:, 0].numpy().tolist(),
            "frame": idcs[:, 1].numpy().tolist(),
            "shape": [shape] * len(idcs[:, 1]),
            "center": center.cpu().tolist() * len(idcs[:, 1]),
            "radius": radius.cpu().tolist() * len(idcs[:, 1]),
        }

        seq_df = pd.DataFrame(seq_data)
        log_df = log_df.append(seq_df, ignore_index=True)
        print(log_df)

    log_df.to_pickle(
        "/media/martin/Samsung_T5/data/endoscopic_data/cholec80_circle_tracking/circle_log.pkl"
    )
    log_df.to_csv(
        "/media/martin/Samsung_T5/data/endoscopic_data/cholec80_circle_tracking/circle_log.csv"
    )
