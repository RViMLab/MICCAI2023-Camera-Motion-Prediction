#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda env update -f /home/martin/proj/homography_imitation_learning/env.yml
conda activate hil

python /home/martin/proj/homography_imitation_learning/main_homography_regression.py \
  --server rvim_server \
  --config config/unsupervised_deep_homography_estimation_backbone.yml
