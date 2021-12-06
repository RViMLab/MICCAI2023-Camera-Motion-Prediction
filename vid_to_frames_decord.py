import os
import argparse
import torch
import pandas as pd
from torch.utils import data
from tqdm import tqdm
from decord import VideoLoader, cpu, bridge

import endoscopy
from datasets import DecordDataset
from torch.utils.data import DataLoader
from utils.io import recursive_scan2df, load_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sf", "--servers_file", type=str, default="config/servers.yml", help="Servers file.")
    parser.add_argument("-s", "--server", type=str, default="local", help="Specify server.")
    parser.add_argument("-rf", "--recursive_folder", type=str, default="cholec80/videos", help="Folder to be recursively searched, relative to server['database']['location'].")
    parser.add_argument("-of", "--output_folder", type=str, default="cholec80_circle_tracking", help="Output folder, relative to server['database']['location'].")
    parser.add_argument("--seq_len", type=int, default=100, help="Set sequence length used for segmentation reduction.")
    parser.add_argument("--num_threads", type=int, default=8, help="Decord threads for loading frames from video.")
    parser.add_argument("--shape", nargs="+", default=[100, 320, 640, 3], help="Reshaped image shape.")
    args = parser.parse_args()

    server = args.server
    server = load_yaml(args.servers_file)[server]
    prefix = os.path.join(server["database"]["location"], args.recursive_folder)

    data_df = recursive_scan2df(prefix, ".mp4")
    data_df = data_df.sort_values(["folder", "file"]).reset_index(drop=True)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    detector = endoscopy.BoundingCircleDetector(device=device)

    # Create video loader
    paths = [os.path.join(prefix, row.folder, row.file) for _, row in data_df.iterrows()]   
    bridge.set_bridge("torch")
    dl = VideoLoader(
        uris=paths,
        ctx=cpu(0),
        shape=args.shape,
        interval=0,
        skip=0,
        shuffle=0
    )
    log_df = pd.DataFrame(columns=["vid", "frame", "center", "radius", "shape"])

    for batch in tqdm(dl):
        imgs, idcs = batch
        vid_idcs, frame_idcs = idcs[:,0], idcs[:,1]
        imgs = imgs.to(device).float().permute(0, 3, 1, 2)/255.
        center, radius = detector(imgs, reduction="max")

        seq_data = {
            "vid": vid_idcs[0].numpy().tolist(),
            "frame": frame_idcs[0].numpy().tolist(),
            "center": center.cpu().tolist()*len(frame_idcs),
            "radius": radius.cpu().tolist()*len(frame_idcs),
            "shape": [args.shape]*len(frame_idcs)
        }

        seq_df = pd.DataFrame(seq_data)
        log_df = log_df.append(
            seq_df, ignore_index=True
        )

    output_prefix = os.path.join(server["database"]["location"], args.output_folder)
    log_df.to_pickle(os.path.join(output_prefix, "circle_log.pkl"))
    log_df.to_csv(os.path.join(output_prefix, "circle_log.csv"))
