#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda env update -f /workspace/homography_imitation_learning/env.yml
conda activate hil

python /workspace/homography_imitation_learning/vid_to_png.py \
    -s dgx1-1 \
    -o /nfs/mhuber/data/camera_motion_separated_png
