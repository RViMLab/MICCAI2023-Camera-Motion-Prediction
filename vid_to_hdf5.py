import argparse
import os

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import utils
from utils.io import generate_path, load_yaml
from utils.sampling import ConsecutiveSequences
from utils.transforms import any_dict_list_to_compose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--server", type=str, required=True)
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="",
        help="Prefix within database, e.g. camera_motion_separated.",
    )
    parser.add_argument(
        "-d",
        "--dataframe",
        type=str,
        default="config/high_fps_without_camera_motion_videos_transforms.pkl",
    )
    parser.add_argument("-o", "--output_folder", type=str, required=True)
    parser.add_argument("-l", "--log", type=str, default="log_without_camera_motion")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_seq", type=int, default=None)
    parser.add_argument("--seq_stride", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=1)
    args = parser.parse_args()

    servers = load_yaml("config/servers.yml")
    server = servers[args.server]

    # dict of videos
    df = pd.read_pickle(args.dataframe)

    # generate sequence iterator with pre transforms
    if not args.prefix:
        paths = [
            os.path.join(
                server["database"]["location"],
                row.database,
                row.file["path"],
                row.file["name"],
            )
            for _, row in df.iterrows()
        ]
    else:
        paths = [
            os.path.join(
                server["database"]["location"],
                args.prefix,
                row.file["path"],
                row.file["name"],
            )
            for _, row in df.iterrows()
        ]

    composes = [
        any_dict_list_to_compose(row.pre_transforms) for _, row in df.iterrows()
    ]
    consecutive_sequence = ConsecutiveSequences(
        paths=paths,
        stride=args.stride,
        max_seq=args.max_seq,
        seq_len=args.seq_len,
        transforms=composes,
        verbose=True,
    )  # generate iterator

    l = len(consecutive_sequence)
    shape = next(consecutive_sequence)[0][0].shape
    consecutive_sequence.reset()
    print("Total sequence length: {}, single image shape: {}".format(l, shape))

    # generate hdf5 storage
    generate_path(args.output_folder)
    hdf_storage = h5py.File(
        os.path.join(args.output_folder, "{}.hdf5".format(args.log)), "w"
    )
    hdf_storage.create_dataset("img", shape=((l,) + shape), dtype=np.uint8)

    # generate dataframe for meta information
    log_df = pd.DataFrame(columns=["vid", "frame", "test"])

    print("Writing files to HDF5 storage...")
    for idx, (cs, vid_idx, frame_idx) in enumerate(tqdm(consecutive_sequence)):
        hdf_storage["img"][idx] = cs[
            0
        ]  # note: currenlty only saves first image of sequence

        log_df = log_df.append(
            {"vid": vid_idx, "frame": frame_idx, "test": not df.iloc[vid_idx].train},
            ignore_index=True,
        )
    print("Done.")

    hdf_storage.close()
    log_df.to_pickle(os.path.join(args.output_folder, "{}.pkl".format(args.log)))
    log_df.to_csv(os.path.join(args.output_folder, "{}.csv".format(args.log)))


if __name__ == "__main__":
    main()
