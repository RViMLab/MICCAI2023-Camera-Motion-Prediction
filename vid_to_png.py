import cv2
import os
import argparse
import pandas as pd
from tqdm import tqdm

import utils
from utils.io import load_yaml, generate_path
from utils.transforms import Compose, dict_list_to_compose
from utils.sampling import ConsecutiveSequences
from utils.processing import RandomEdgeHomography


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--server', type=str, required=True)
    parser.add_argument('-d', '--databases', type=str, default='configs/high_fps_without_camera_motion_videos_transforms.yml')
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-l', '--log', type=str, default='log_with_camera_motion')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--max_seq', type=int, default=None)
    parser.add_argument('--seq_stride', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=1)
    args = parser.parse_args()

    servers = load_yaml('configs/servers.yml')
    server = args.server

    # dict of videos
    yml = load_yaml(path=args.databases)

    absolute_prefix = args.output_folder
    generate_path(absolute_prefix)

    df = pd.DataFrame(columns=['file', 'path', 'vid_idx', 'frame_idx', 'database'])

    for database in yml['databases']:

        print('Processing database {}'.format(database['name']))
        paths = []
        for files in database['videos']['files']:
            paths.append(os.path.join(
                servers[server]['database']['location'], 
                database['prefix'], 
                database['videos']['prefix'], 
                files)
            )

        composes = []
        for dict_list in database['transforms']:
            composes.append(dict_list_to_compose(dict_list, utils.transforms)) # generate compose transform from dict
        consecutive_sequences = ConsecutiveSequences(
            paths=paths, 
            stride=args.stride, 
            max_seq=args.max_seq, 
            seq_stride=args.seq_stride, 
            seq_len=args.seq_len, 
            transforms=composes, 
            verbose=True
        ) # generate iterator
        
        for cs, vid_idx, frame_idx in tqdm(consecutive_sequences):
            # only safe first image of sequence
            if database['test'] == True:
                relative_prefix = 'test'
            else:
                relative_prefix = 'train'
            relative_prefix = os.path.join(relative_prefix, database['videos']['files'][vid_idx][:-4]) # unique name of the video

            file_name = 'frame_{}.png'.format(frame_idx)

            prefix = os.path.join(absolute_prefix, relative_prefix)
            generate_path(prefix)

            path = os.path.join(prefix, file_name)
            cv2.imwrite(path, cs[0]) # note: currenlty only safes first image of sequence

            df = df.append({
                'file': file_name, 
                'path': relative_prefix, 
                'vid_idx': vid_idx, 
                'frame_idx': frame_idx,
                'database': {'name': database['name'], 'test': database['test']}
            }, ignore_index=True)
                
    df_name = args.log
    df.to_pickle(os.path.join(absolute_prefix, '{}.pkl'.format(df_name)))
    df.to_csv(os.path.join(absolute_prefix, '{}.csv'.format(df_name)))
