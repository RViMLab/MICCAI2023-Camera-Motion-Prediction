import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.io import generate_path, load_yaml
from utils.processing import RandomEdgeHomography
from utils.sampling import ConsecutiveSequences
from utils.transforms import Compose, any_dict_list_to_compose


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
    log_df = pd.DataFrame(columns=["file", "path", "vid_idx", "frame_idx", "database"])

    output_prefix = args.output_folder
    generate_path(output_prefix)

    for row_idx, row in df.iterrows():
        print("Processing row {}/{}".format(row_idx, len(df)))
        if not args.prefix:
            paths = [
                os.path.join(
                    server["database"]["location"],
                    row.database,
                    row.file["path"],
                    row.file["name"],
                )
            ]
        else:
            paths = [
                os.path.join(
                    server["database"]["location"],
                    args.prefix,
                    row.file["path"],
                    row.file["name"],
                )
            ]

        vid_relative_output_prefix = os.path.join(
            row.file["path"], row.file["name"][:-4]
        )
        vid_absolute_output_prefix = os.path.join(
            output_prefix, vid_relative_output_prefix
        )
        generate_path(vid_absolute_output_prefix)

        compose = [any_dict_list_to_compose(row.pre_transforms)]
        consecutive_sequence = ConsecutiveSequences(
            paths=paths,
            stride=args.stride,
            max_seq=args.max_seq,
            seq_len=args.seq_len,
            transforms=compose,
            verbose=True,
        )  # generate iterator

        for cs, vid_idx, frame_idx in tqdm(consecutive_sequence):
            file_name = "frame_{}.npy".format(frame_idx)
            np.save(
                os.path.join(vid_absolute_output_prefix, file_name), cs[0]
            )  # note: currenlty only saves first image of sequence

            log_df = log_df.append(
                {
                    "file": file_name,
                    "path": vid_relative_output_prefix,
                    "vid_idx": vid_idx,
                    "frame_idx": frame_idx,
                    "database": {"name": row.database, "test": not row.train},
                },
                ignore_index=True,
            )  # legacy

    df_name = args.log
    log_df.to_pickle(os.path.join(output_prefix, "{}.pkl".format(df_name)))
    log_df.to_csv(os.path.join(output_prefix, "{}.csv".format(df_name)))


if __name__ == "__main__":
    main()
