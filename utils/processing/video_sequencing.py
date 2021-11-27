import os
import cv2
import torch
import pathlib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from multiprocessing import Pool, Value
from kornia import tensor_to_image, image_to_tensor
from kornia.geometry import crop_and_resize

from utils.io import recursive_scan2df
from endoscopy import BoundingCircleDetector, max_rectangle_in_circle


processed_cnt = Value('i', 0)

class MultiProcessVideoSequencer(object):

    def __init__(self, prefix: str, postfix: str=".mp4", shape: Tuple[int, int]=(240, 320), buffer_size: int=10) -> None:
        self._prefix = prefix
        self._postfix = postfix
        self._shape = shape
        self._buffer_size = buffer_size

        self._df = recursive_scan2df(self._prefix, postfix=self._postfix)
        self._df = self._df.sort_values("file").reset_index(drop=True)
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
        log_df = log_df.sort_values(["vid", "frame"]).reset_index(drop=True)
        log_df.to_pickle("{}/log.pkl".format(output_prefix))
        log_df.to_csv("{}/log.csv".format(output_prefix))

    def process_buffer(self, buffer: List[Dict]):
        for element in buffer:
            element["img"] = cv2.resize(element["img"], (self._shape[1], self._shape[0]), interpolation=cv2.INTER_CUBIC)[...,::-1]
        return buffer

    def empty_buffer(self, buffer: List[Dict], log_df: pd.DataFrame) -> pd.DataFrame:
        buffer = self.process_buffer(buffer)

        for element in buffer:
            file = "frame_{}.npy".format(element["frame_cnt"])
            np.save(os.path.join(element["path"], file), element["img"])

            # log
            log_df = log_df.append({
                "folder": element["folder"],
                "file": file,
                "vid": element["vid_idx"],
                "frame": element["frame_cnt"]
            }, ignore_index=True)

        buffer.clear()
        return log_df

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
        buffer = []

        while success:
            buffer.append({
                "img": img,
                "path": path,
                "folder": folder,
                "vid_idx": vid_idx,
                "frame_cnt": frame_cnt
            })

            if len(buffer) >= self._buffer_size:
                log_df = self.empty_buffer(buffer, log_df)
            success, img = vc.read()
            frame_cnt += 1

        vc.release()

        # empty at end
        log_df = self.empty_buffer(buffer, log_df)

        # output progress
        global processed_cnt
        with processed_cnt.get_lock():
            processed_cnt.value += 1
        print("Processed {}/{}.\r".format(processed_cnt.value, len(self._df)), end="")

        return log_df


class SingleProcessInferenceVideoSequencer():
    def __init__(self, prefix: str, postfix: str=".mp4", shape: Tuple[int]=(240, 320), seq_len: int=25, buffer_size: int=10) -> None:
        self._prefix = prefix
        self._postfix = postfix
        self._shape = shape
        self._seq_len = seq_len
        self._buffer_size = buffer_size
        self._detector = BoundingCircleDetector()

        self._df = recursive_scan2df(self._prefix, postfix=self._postfix)
        self._df = self._df.sort_values("file").reset_index(drop=True)
        print("Loaded videos:\n", self._df)

    def process_buffer(self, buffer: List[Dict]):
        imgs = []
        for element in buffer:
            imgs.append(image_to_tensor(element["img"], True))
        imgs = torch.stack(imgs, axis=0)
        
        # normalize
        imgs = imgs.float()/255.

        # inference
        center, radius = self._detector(imgs, N=1000)
        box = max_rectangle_in_circle(imgs.shape, center, radius)
        imgs = crop_and_resize(imgs, box, self._shape)

        # update buffer
        for idx, img in enumerate(imgs):
            buffer[idx]["img"] = tensor_to_image(img, False)

        return buffer

    def empty_buffer(self, buffer: List[Dict], log_df: pd.DataFrame) -> pd.DataFrame:
        buffer = self.process_buffer(buffer)

        for element in buffer:
            file = "frame_{}.npy".format(element["frame_cnt"])
            np.save(os.path.join(element["path"], file), element["img"])

            # log
            log_df = log_df.append({
                "folder": element["folder"],
                "file": file,
                "vid": element["vid_idx"],
                "frame": element["frame_cnt"]
            }, ignore_index=True)

        buffer.clear()
        return log_df

    def start(self, output_prefix: str) -> None:
        log_df = pd.DataFrame(columns=["folder", "file", "vid", "frame"])

        # for each video
        for idx, row in self._df.iterrows():
            vid_idx = idx
            video_path = os.path.join(self._prefix, row.folder, row.file)

            vc = cv2.VideoCapture(video_path)

            folder = "vid_{}".format(vid_idx)
            path = pathlib.Path(os.path.join(output_prefix, folder))
            if not path.exists():
                path.mkdir(parents=True)

            success, img = vc.read()
            frame_cnt = 0
            buffer = []

            while success:
                buffer.append({
                    "img": img,
                    "path": path,
                    "folder": folder,
                    "vid_idx": vid_idx,
                    "frame_cnt": frame_cnt
                })

                if len(buffer) >= self._buffer_size:
                    log_df = self.empty_buffer(buffer, log_df)
                success, img = vc.read()
                frame_cnt += 1

            vc.release()

            # empty at end
            log_df = self.empty_buffer(buffer, log_df)   

        log_df = log_df.sort_values(["vid", "frame"]).reset_index(drop=True)
        log_df.to_pickle("{}/log.pkl".format(output_prefix))
        log_df.to_csv("{}/log.csv".format(output_prefix))


if __name__ == "__main__":
    from utils.io import load_yaml

    server = "local"
    server = load_yaml("config/servers.yml")[server]
    prefix = os.path.join(server["database"]["location"], "cholec80/sample_videos")

    vs = MultiProcessVideoSequencer(
        prefix=prefix, 
        postfix=".mp4",
        shape=(240, 320)
    )

    output_prefix = os.path.join(server["database"]["location"], "cholec80_frames")
    vs.start(output_prefix=output_prefix, processes=4)
