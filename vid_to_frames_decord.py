import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm

import endoscopy
from datasets import DecordDataset
from torch.utils.data import DataLoader
from utils.io import recursive_scan2df, load_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sf", "--servers_file", type=str, default="config/servers.yml", help="Servers file.")
    parser.add_argument("-s", "--server", type=str, default="local", help="Specify server.")
    parser.add_argument("-rf", "--recursive_folder", type=str, default="cholec80/sample_videos", help="Folder to be recursively searched, relative to server['database']['location'].")
    parser.add_argument("-of", "--output_folder", type=str, default="cholec80_circle_tracking", help="Output folder, relative to server['database']['location'].")
    parser.add_argument("--seq_len", type=int, default=100, help="Set sequence length used for segmentation reduction.")
    parser.add_argument("--num_threads", type=int, default=5, help="Decord threads for loading frames from video.")
    args = parser.parse_args()

    server = args.server
    server = load_yaml(args.servers_file)[server]
    prefix = os.path.join(server["database"]["location"], args.recursive_folder)

    data_df = recursive_scan2df(prefix, ".mp4")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    detector = endoscopy.BoundingCircleDetector(device=device)

    ds = DecordDataset(
        data_df=data_df,
        prefix=prefix,
        seq_len=args.seq_len,
        frame_increment=1,
        frames_between_clips=args.seq_len,
        num_threads=args.num_threads
    )
    dl = DataLoader(ds, batch_size=1, num_workers=0)
    log_df = pd.DataFrame(columns=["vid", "frame", "center", "radius"])

    for batch in tqdm(dl):
        imgs, frame_idcs, vid_idcs = batch
        imgs = imgs[0].to(device).float().permute(0, 3, 1, 2)/255.
        center, radius = detector(imgs, reduction="max")

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

    output_prefix = os.path.join(server["database"]["location"], args.output_folder)
    log_df.to_pickle(os.path.join(output_prefix, "circle_log.pkl"))
    log_df.to_csv(os.path.join(output_prefix, "circle_log.csv"))
