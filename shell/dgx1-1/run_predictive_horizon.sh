#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate torch110

python /workspace/homography_imitation_learning/homography_imitation_seq_main.py \
  --server dgx1-1 \
  --config config/predictive_horizon_seq_dgx.yml \
  --backbone_path ae_cai/resnet/48/25/34/version_0
