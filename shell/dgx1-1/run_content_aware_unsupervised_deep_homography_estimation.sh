#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate torch110

python /workspace/homography_imitation_learning/homography_regression_main.py \
  --server dgx1-1 \
  --config config/content_aware_unsupervised_deep_homography_estimation.yml
