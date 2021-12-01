import os
import argparse

from utils.processing import MultiProcessVideoSequencer, SingleProcessInferenceVideoSequencer
from utils.io import load_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sf", "--servers_file", type=str, default="config/servers.yml", help="Servers file.")
    parser.add_argument("-s", "--server", type=str, default="local", help="Specify server.")
    parser.add_argument("-rf", "--recursive_folder", type=str, default="cholec80/sample_videos", help="Folder to be recursively searched, relative to server['database']['location'].")
    parser.add_argument("-of", "--output_folder", type=str, default="cholec80_frames", help="Output folder, relative to server['database']['location'].")
    parser.add_argument("-p", "--processes", type=int, default=4, help="Number of processes to sequence videos.")
    parser.add_argument("-mp", "--multiprocessed", type=bool, default=False, help="Multiprocessed sequencing or single processed with inference.")
    args = parser.parse_args()

    server = args.server
    server = load_yaml(args.servers_file)[server]
    prefix = os.path.join(server["database"]["location"], args.recursive_folder)

    if args.multiprocessed:
        vs = MultiProcessVideoSequencer(
            prefix=prefix, 
            postfix=".mp4",
            shape=(240, 320),
            buffer_size=20
        )
    else:
        vs = SingleProcessInferenceVideoSequencer(
            prefix=prefix,
            postfix=".mp4",
            shape=(240, 320),
            batch_size=20,
            sequence_length=25
        )

    output_prefix = os.path.join(server["database"]["location"], args.output_folder)
    print("Writing files to {}...".format(output_prefix))

    if args.multiprocessed:
        vs.start(output_prefix=output_prefix, processes=args.processes)
    else:
        vs.start(output_prefix=output_prefix)
    print("\nDone.")
