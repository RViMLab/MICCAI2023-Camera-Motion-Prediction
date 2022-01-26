#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate torch110

python /workspace/homography_imitation_learning/homography_imitation_seq_main.py \
  --server dgx1-1 \
  --config config/duv_lstm.yml
