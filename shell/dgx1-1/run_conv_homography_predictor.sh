#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate torch110

python /workspace/homography_imitation_learning/main_homography_imitation_seq.py \
  --server dgx1-1 \
  --config config/conv_homography_predictor_phantom.yml \
  --backbone_path ae_cai/resnet/48/25/34/version_0