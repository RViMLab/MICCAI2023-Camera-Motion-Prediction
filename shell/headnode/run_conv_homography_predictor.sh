#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate torch110

python /workspace/homography_imitation_learning/main_homography_imitation_seq.py \
  --server headnode \
  --config config/conv_homography_predictor_phantom.yml \
  --homography_regression ae_cai/resnet/48/25/34/version_0
