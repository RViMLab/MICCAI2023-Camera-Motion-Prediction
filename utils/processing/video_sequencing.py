import os
import cv2
import pathlib
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from multiprocessing import Pool, Value

from utils.io import recursive_scan2df


processed_cnt = Value('i', 0)

class VideoSequencer(object):

    def __init__(self, prefix: str, postfix: str=".mp4", shape: Tuple[int, int]=(240, 320), interpolation: int=cv2.INTER_CUBIC) -> None:
        self._prefix = prefix
        self._postfix = postfix
        self._shape = shape
        self._interpolation = interpolation

        self._df = recursive_scan2df(self._prefix, postfix=self._postfix)
        self._df = self._df.sort_values("file").reset_index()
        print("Loaded videos:\n", self._df)

    def start(self, output_prefix: str, processes: int=2) -> None:
 
        kwargs_list = [
            {
                "vid_idx": idx, 
                "video_path": os.path.join(self._prefix, row.folder, row.file), 
                "output_prefix": output_prefix
            } for idx, row in self._df.iterrows()
        ]

        p = Pool(processes=processes)
        log_df_list = p.map(
            self._sequencing_thread, kwargs_list
        )

        log_df = pd.concat(log_df_list)
        log_df = log_df.sort_values(["vid", "frame"])
        log_df.to_pickle("{}/log.pkl".format(output_prefix))
        log_df.to_csv("{}/log.csv".format(output_prefix))

    def _sequencing_thread(self, dict: Dict):
        vid_idx = dict["vid_idx"]
        video_path = dict["video_path"]
        output_prefix = dict["output_prefix"]

        vc = cv2.VideoCapture(video_path)
        log_df = pd.DataFrame(columns=["folder", "file", "vid", "frame"])

        folder = "vid_{}".format(vid_idx)
        path = pathlib.Path(os.path.join(output_prefix, folder))
        if not path.exists():
            path.mkdir(parents=True)

        success, img = vc.read()
        frame_cnt = 0
        while success:
            img = cv2.resize(img, (self._shape[1], self._shape[0]), interpolation=self._interpolation)[...,::-1]
            file = "frame_{}.npy".format(frame_cnt)
            np.save(os.path.join(path, file), img)

            # log
            log_df = log_df.append({
                "folder": folder,
                "file": file,
                "vid": vid_idx,
                "frame": frame_cnt
            }, ignore_index=True)

            success, img = vc.read()
            frame_cnt += 1
        vc.release()

        # output progress
        global processed_cnt
        with processed_cnt.get_lock():
            processed_cnt.value += 1
        print("Processed {}/{}.\r".format(processed_cnt.value, len(self._df)), end="")

        return log_df


if __name__ == "__main__":
    from utils.io import load_yaml

    server = "local"
    server = load_yaml("config/servers.yml")[server]
    prefix = os.path.join(server["database"]["location"], "cholec80/sample_videos")

    vs = VideoSequencer(
        prefix=prefix, 
        postfix=".mp4",
        shape=(240, 320)
    )

    output_prefix = os.path.join(server["database"]["location"], "cholec80_frames")
    vs.start(output_prefix=output_prefix, processes=4)
