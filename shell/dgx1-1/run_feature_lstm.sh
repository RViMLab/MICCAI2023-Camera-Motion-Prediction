#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate torch110

python /workspace/homography_imitation_learning/main_homography_imitation_seq.py \
  --server dgx1-1 \
  --config config/feature_lstm.yml
