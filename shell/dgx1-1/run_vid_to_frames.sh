#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda env update -f /workspace/homography_imitation_learning/env_torch110.yml
conda activate torch110

python /workspace/homography_imitation_learning/vid_to_frames.py \
  --servers_file config/servers.yml \
  --server dgx1-1 \
  --recursive_folder cholec80/videos \
  --output_folder cholec80_frames \
  --processes 20
