import os
import argparse

from utils.processing import VideoSequencer
from utils.io import load_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sf", "--servers_file", type=str, default="config/servers.yml", help="Servers file.")
    parser.add_argument("-s", "--server", type=str, default="local", help="Specify server.")
    parser.add_argument("-rf", "--recursive_folder", type=str, default="cholec80/sample_videos", help="Folder to be recursively searched, relative to server['database']['location'].")
    parser.add_argument("-of", "--output_folder", type=str, default="cholec80_frames", help="Output folder, relative to server['database']['location'].")
    parser.add_argument("-p", "--processes", type=int, default=2, help="Number of processes to sequence videos.")
    args = parser.parse_args()

    server = args.server
    server = load_yaml(args.servers_file)[server]
    prefix = os.path.join(server["database"]["location"], args.recursive_folder)

    vs = VideoSequencer(
        prefix=prefix, 
        postfix=".mp4",
        shape=(240, 320)
    )

    output_prefix = os.path.join(server["database"]["location"], args.output_folder)
    print("Writing files to {}...".format(output_prefix))
    vs.start(output_prefix=output_prefix, processes=args.processes)
    print("\nDone.")
