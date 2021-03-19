#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda env update -f /workspace/homography_imitation_learning/env_dgx.yml
conda activate hil

python /workspace/homography_imitation_learning/homography_imitation_main.py \
  --server dgx1-1 \
  --configs config/next_view_dgx.yml \
  --backbone_path deep_image_homography_estimation_backbone/version_2
