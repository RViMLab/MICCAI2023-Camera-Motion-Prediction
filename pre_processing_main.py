import os
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.processing import frame_pairs
from utils.io import load_yaml, scan2df, natural_keys
from lightning_modules import DeepImageHomographyEstimationModuleBackbone
from datasets import ImageSequenceDataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="local", help="Server to be used.")
    parser.add_argument("--servers_file", type=str, default="config/servers.yml", help="Server configuration file.")
    parser.add_argument("--backbone_path", type=str, default="ae_cai/resnet/48/25/34/version_0", help="Path to log folders, relative to server logging location.")
    parser.add_argument("--data_prefix", type=str, default="cholec80_single_video_frames_cropped", help="Relative path to data from database location.")
    parser.add_argument("--in_pkl", type=str, default="log.pkl", help="Pickle file with database information.")
    parser.add_argument("--out_pkl", type=str, default="pre_processed_log.pkl", help="Pickle file with preprocessed information.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for data loading.")
    parser.add_argument("--nth_frame", type=int, default=1, help="Process every nth frame.")
    args = parser.parse_args()

    servers = load_yaml(args.servers_file)
    server = servers[args.server]

    # Load model
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    backbone_path = os.path.join(server["logging"]["location"], args.backbone_path)
    backbone_configs = load_yaml(os.path.join(server["logging"]["location"], args.backbone_path, "config.yml"))
    backbone_configs["model"]["pretrained"] = False  # set to false, as loaded anyways
    df = scan2df(os.path.join(server["logging"]["location"], args.backbone_path, "checkpoints"), ".ckpt")
    ckpts = sorted(list(df["file"]), key=natural_keys)

    module = DeepImageHomographyEstimationModuleBackbone.load_from_checkpoint(
        checkpoint_path=os.path.join(backbone_path, "checkpoints/{}".format(ckpts[-1])),
        **backbone_configs["model"]
    )
    module = module.eval().to(device)
    module.freeze()

    # Prepare data
    data_prefix = os.path.join(server["database"]["location"], args.data_prefix)
    df = pd.read_pickle(os.path.join(data_prefix, args.in_pkl))

    ds = ImageSequenceDataset(
        df, data_prefix, seq_len=2, frame_increment=args.nth_frame, frames_between_clips=1
    )
    ds._df = ds._df.astype(object)
    dl = DataLoader(ds, num_workers=args.num_workers, batch_size=args.batch_size, drop_last=False, shuffle=False)

    # Prepase logging data frame
    duv_df = pd.DataFrame({"duv": np.full(len(df), np.nan)})
    duv_df = duv_df.astype(object)

    for vid, tf_vid, idcs, vid_idcs in tqdm(dl):
        vid = vid.to(device=device, dtype=torch.float)/255.  # without shuffling, average circle detection over whole video

        frames_i, frames_ips = frame_pairs(vid, 1)  # re-sort images
        frames_i   = frames_i.reshape((-1,) + frames_i.shape[-3:])      # reshape BxNxCxHxW -> B*NxCxHxW
        frames_ips = frames_ips.reshape((-1,) + frames_ips.shape[-3:])  # reshape BxNxCxHxW -> B*NxCxHxW

        duvs = module(frames_i, frames_ips)

        idcs = idcs[:,0].view(-1).numpy()  # take starting index
        duvs = duvs.cpu().numpy()

        for cnt, idx in enumerate(idcs):
            duv_df.loc[idx].duv = duvs[cnt].tolist()

    df["duv"] = duv_df.duv

    # Compute mean motion
    df['duv_mpd'] = df.duv.apply(lambda x:
        x if np.isnan(x).any() else
        np.linalg.norm(x, axis=1).mean()
    )

    # Safe data
    df.to_pickle(os.path.join(data_prefix, args.out_pkl))
