#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate torch110

python /workspace/homography_imitation_learning/vid_to_frames.py \
  --servers_file config/servers.yml \
  --server dgx1-1 \
  --recursive_folder cholec80/sample_videos \
  --output_folder cholec80_single_video_frames \
  --shape 100 480 640 3
