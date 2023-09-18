#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate torch110

python /workspace/homography_imitation_learning/main_autoencoder.py \
  --server headnode \
  --config autoencoder.yml
