import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from decord import VideoLoader, cpu, bridge

import endoscopy
from utils.io import recursive_scan2df, load_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sf", "--servers_file", type=str, default="config/servers.yml", help="Servers file.")
    parser.add_argument("-s", "--server", type=str, default="local", help="Specify server.")
    parser.add_argument("-rf", "--recursive_folder", type=str, default="cholec80/videos", help="Folder to be recursively searched, relative to server['database']['location'].")
    parser.add_argument("-of", "--output_folder", type=str, default="cholec80_circle_tracking_mean_reduction", help="Output folder, relative to server['database']['location'].")
    parser.add_argument("-r", "--reduction", type=str, default="mean", help="Reduction to be applied to segmented image sequence.")
    parser.add_argument("--shape", nargs="+", default=[100, 480, 640, 3], help="Reshaped image shape BxHxWxC, C=3.")
    args = parser.parse_args()

    server = args.server
    server = load_yaml(args.servers_file)[server]
    prefix = os.path.join(server["database"]["location"], args.recursive_folder)

    data_df = recursive_scan2df(prefix, ".mp4")
    data_df = data_df.sort_values(["folder", "file"]).reset_index(drop=True)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    detector = endoscopy.BoundingCircleDetector(device=device, model_enum=endoscopy.SEGMENTATION_MODEL.UNET_RESNET_34_TINY)

    # Create video loader
    paths = [os.path.join(prefix, row.folder, row.file) for _, row in data_df.iterrows()]
    bridge.set_bridge("torch")

    for path in paths:
        vid_idx = int(path.split("/")[-1].split(".")[0][-2:]) - 1
        print(vid_idx)
        
        dl = VideoLoader(
            uris=[path],
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
            try:
                center, radius = detector(imgs, reduction=args.reduction)
                if args.reduction is not None:
                    seq_data = {
                        "vid": [vid_idx]*len(frame_idcs),
                        "frame": frame_idcs.numpy().tolist(),
                        "center": center.cpu().tolist()*len(frame_idcs),
                        "radius": radius.cpu().tolist()*len(frame_idcs),
                        "shape": [args.shape]*len(frame_idcs)
                    }
                else:
                    seq_data = {
                        "vid": [vid_idx]*len(frame_idcs),
                        "frame": frame_idcs.numpy().tolist(),
                        "center": center.cpu().tolist(),
                        "radius": radius.cpu().tolist(),
                        "shape": [args.shape]*len(frame_idcs)
                    }
            except:
                center, radius = torch.full([len(frame_idcs), 2], float('nan')), torch.full([len(frame_idcs)], float('nan'))

                seq_data = {
                    "vid": [vid_idx]*len(frame_idcs),
                    "frame": frame_idcs.numpy().tolist(),
                    "center": center.cpu().tolist(),
                    "radius": radius.cpu().tolist(),
                    "shape": [args.shape]*len(frame_idcs)
                }

            seq_df = pd.DataFrame(seq_data)
            log_df = log_df.append(
                seq_df, ignore_index=True
            )

        del dl

        output_prefix = os.path.join(server["database"]["location"], args.output_folder)
        log_df.to_pickle(os.path.join(output_prefix, "circle_log_{}.pkl".format(vid_idx)))
        log_df.to_csv(os.path.join(output_prefix, "circle_log_{}.csv".format(vid_idx)))
