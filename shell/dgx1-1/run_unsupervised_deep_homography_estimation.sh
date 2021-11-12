#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda env update -f /workspace/homography_imitation_learning/env_dgx.yml
conda activate torch110

python /workspace/homography_imitation_learning/homography_regression_main.py \
  --server dgx1-1 \
  --config config/unsupervised_deep_homography_estimation.yml
