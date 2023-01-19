import argparse
import os

import pandas as pd

from utils.io import load_yaml
from utils.processing import (
    MultiProcessVideoSequencer,
    MultiProcessVideoSequencerPlusCircleCropping,
    SingleProcessInferenceVideoSequencer,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sf",
        "--servers_file",
        type=str,
        default="config/servers.yml",
        help="Servers file.",
    )
    parser.add_argument(
        "-s", "--server", type=str, default="local", help="Specify server."
    )
    parser.add_argument(
        "-rf",
        "--recursive_folder",
        type=str,
        default="cholec80/sample_videos",
        help="Folder to be recursively searched, relative to server['database']['location'].",
    )
    parser.add_argument(
        "-of",
        "--output_folder",
        type=str,
        default="cholec80_test",
        help="Output folder, relative to server['database']['location'].",
    )
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=4,
        help="Number of processes to sequence videos.",
    )
    parser.add_argument(
        "-mp",
        "--multiprocessed",
        type=bool,
        default=False,
        help="Multiprocessed sequencing or single processed with inference.",
    )
    parser.add_argument(
        "-pp",
        "--pre_process",
        type=bool,
        default=False,
        help="Pre-process frames in a multiprocess fashion.",
    )
    parser.add_argument(
        "-cf",
        "--circle_file",
        type=str,
        default="cholec80_circle_tracking_individual/df_interpolated/log.pkl",
        help="Circle cropping information file, relative to server['database']['location'].",
    )
    args = parser.parse_args()

    server = args.server
    server = load_yaml(args.servers_file)[server]
    prefix = os.path.join(server["database"]["location"], args.recursive_folder)

    if args.multiprocessed:
        if args.pre_process:
            circle_df = pd.read_pickle(
                os.path.join(server["database"]["location"], args.circle_file)
            )

            vs = MultiProcessVideoSequencerPlusCircleCropping(
                prefix=prefix,
                circle_df=circle_df,
                postfix=".mp4",
                shape=(240, 320),
                buffer_size=20,
            )
        else:
            vs = (
                MultiProcessVideoSequencer(  # scaled crop. Discard non-existing indices
                    prefix=prefix, postfix=".mp4", shape=(240, 320), buffer_size=20
                )
            )
    else:
        vs = SingleProcessInferenceVideoSequencer(
            prefix=prefix,
            postfix=".mp4",
            shape=(240, 320),
            batch_size=20,
            sequence_length=25,
        )

    output_prefix = os.path.join(server["database"]["location"], args.output_folder)
    print("Writing files to {}...".format(output_prefix))

    if args.multiprocessed:
        vs.start(output_prefix=output_prefix, processes=args.processes)
    else:
        vs.start(output_prefix=output_prefix)
    print("\nDone.")
