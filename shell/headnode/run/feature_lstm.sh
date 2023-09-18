#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate hil_torch110

python /workspace/homography_imitation_learning/main_homography_imitation_seq.py \
  --server headnode \
  --config config/homography_imitation/feature_lstm.yml
