#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate torch110

python /workspace/homography_imitation_learning/main_homography_regression.py \
  --server headnode \
  --config config/deep_image_homography_estimation_backbone.yml
