#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate torch110

python /workspace/homography_imitation_learning/homography_regression_main.py \
  --server dgx1-1 \
  --config config/deep_image_homography_estimation_backbone_increase.yml
