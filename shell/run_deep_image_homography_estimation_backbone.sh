#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda env update -f /workspace/homography_imitation_learning/env.yml
conda activate hil

python /workspace/homography_imitation_learning/homography_regression_main.py \
  --server rvim_server \
  --configs configs/deep_image_homography_estimation_backbone.yml
