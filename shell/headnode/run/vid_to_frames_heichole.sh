#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate torch110

python /workspace/homography_imitation_learning/vid_to_frames.py \
  --servers_file config/servers.yml \
  --server headnode \
  --recursive_folder heichole/Videos/Full_SD \
  --output_folder heichole_single_video_frames_cropped \
  --processes 6 \
  --multiprocessed true \
  --pre_process true \
  --circle_file heichole_circle_tracking_individual/df_interpolated/log.pkl
