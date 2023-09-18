#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate torch110

python /workspace/homography_imitation_learning/main_homography_imitation_seq.py \
  --server headnode \
  --config config/homography_imitation/motion_labels/conv_homography_predictor_cholec80.yml \
  --homography_regression ae_cai/resnet/48/25/34/version_0
