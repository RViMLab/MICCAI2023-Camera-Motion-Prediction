#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda env update -f /workspace/homography_imitation_learning/env.yml
conda activate hil

python /workspace/homography_imitation_learning/homography_imitation_main.py \
  --server dgx1-1 \
  --configs configs/next_view.yml \
  --backbone_path deep_image_homography_estimation_backbone/version_2
