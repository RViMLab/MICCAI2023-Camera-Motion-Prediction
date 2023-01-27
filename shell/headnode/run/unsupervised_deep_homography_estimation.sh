#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate torch113

python /workspace/homography_imitation_learning/main_homography_regression.py \
  --server headnode \
  --config config/unsupervised_deep_homography_estimation.yml
