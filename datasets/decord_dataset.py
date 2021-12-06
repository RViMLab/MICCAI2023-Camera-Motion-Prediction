import os
import random
import torch
import pandas as pd
import numpy as np
from decord import VideoReader
from typing import Callable
from torch.utils.data import Dataset


class DecordDataset(Dataset):
    def __init__(self,
        data_df: list,
        prefix: str,
        seq_len: int = 1,
        frame_increment: int=1,
        frames_between_clips: int=1,
        seeds: bool=True,
        transforms: Callable = None,
        num_threads: int = 0
    ):
        self._data_df = data_df.sort_values(["folder", "file"]).reset_index(drop=True)
        self._seq_len = seq_len
        self._frame_increment = frame_increment
        self._frames_between_clips = frames_between_clips
        self._transforms = transforms
        if self._transforms:
            raise NotImplemented("DecordDataset: No transforms implemented.")
        self._seeds = seeds

        # Create video reader for each file
        self.video_readers = []
        self._vid_df = pd.DataFrame(columns=["vid", "frame"])
        for idx, row in self._data_df.iterrows():
            absolut_path = os.path.join(prefix, row.folder, row.file)
            self.video_readers.append(VideoReader(
                uri=absolut_path,
                num_threads=num_threads
            ))       
            
            frame_count = len(self.video_readers[idx])

            vid_df = pd.DataFrame(columns=["vid", "frame"])
            vid_df["vid"] = [idx]*frame_count
            vid_df["frame"] = [i for i in range(frame_count)]
            self._vid_df = self._vid_df.append(
                vid_df, ignore_index=True
            )

        self._vid_df = self._vid_df.sort_values(["vid", "frame"]).reset_index(drop=True)

        self._idcs = self._filterFeasibleSequenceIndices(
            self._vid_df, col="vid",
            seq_len=self._seq_len,
            frame_increment=self._frame_increment,
            frames_between_clips=self._frames_between_clips
        )

    def __getitem__(self, idx):
        if self._seeds:
            seed = random.randint(0, np.iinfo(np.int32).max)  # set random seeed
        else:
            seed = idx

        frame_idcs = self._vid_df.loc[self._idcs[idx]].frame + np.arange(self._seq_len)*self._frame_increment
        vid_idx = self._vid_df.loc[self._idcs[idx]].vid

        # sample a sequence in video
        sequence = self.video_readers[vid_idx].get_batch(
            frame_idcs
        ).asnumpy()

        # TODO: add transforms

        return sequence, frame_idcs, torch.full(frame_idcs.shape, vid_idx)

    def __len__(self):
        return len(self._idcs)

    def _filterFeasibleSequenceIndices(self, 
        df: pd.DataFrame,
        col: str="vid",
        seq_len: int=2,
        frame_increment: int=1,
        frames_between_clips: int=1

    ) -> pd.DataFrame:
        grouped_df = df.groupby(col)
        return grouped_df.apply(
            lambda x: x.iloc[:len(x) - (seq_len - 1)*frame_increment:frames_between_clips]  # get indices [0, length - (seq_len - 1)]
        ).index.get_level_values(1)  # return 2nd values of pd.MultiIndex


if __name__ == "__main__":
    import torch
    import endoscopy
    from tqdm import tqdm

    from torch.utils.data import DataLoader
    from utils.io import recursive_scan2df

    prefix = "/media/martin/Samsung_T5/data/endoscopic_data/cholec80/sample_videos"
    data_df = recursive_scan2df(prefix, ".mp4")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    detector = endoscopy.BoundingCircleDetector(device=device)

    ds = DecordDataset(
        data_df=data_df,
        prefix=prefix,
        seq_len=100,
        frame_increment=1,
        frames_between_clips=150,
        num_threads=1
    )
    dl = DataLoader(ds, batch_size=1, num_workers=0)
    log_df = pd.DataFrame(columns=["vid", "frame", "center", "radius"])

    for batch in tqdm(dl):
        imgs, frame_idcs, vid_idcs = batch
        imgs = imgs[0].to(device).float().permute(0, 3, 1, 2)/255.
        center, radius = detector(imgs, reduction="max")
        # box = endoscopy.max_rectangle_in_circle(imgs.shape, center, radius)
        # crops = kornia.geometry.crop_and_resize(imgs, box, [240, 320])

        seq_data = {
            "vid": vid_idcs[0].numpy().tolist(),
            "frame": frame_idcs[0].numpy().tolist(),
            "center": center.cpu().tolist()*len(frame_idcs[0]),
            "radius": radius.cpu().tolist()*len(frame_idcs[0])
        }

        seq_df = pd.DataFrame(seq_data)
        log_df = log_df.append(
            seq_df, ignore_index=True
        )

    log_df.to_pickle("/media/martin/Samsung_T5/data/endoscopic_data/cholec80_circle_tracking/circle_log.pkl")
    log_df.to_csv("/media/martin/Samsung_T5/data/endoscopic_data/cholec80_circle_tracking/circle_log.csv")
