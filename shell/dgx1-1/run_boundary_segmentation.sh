#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda env update -f /workspace/homography_imitation_learning/env_torch19.yml
conda activate torch19

python /workspace/homography_imitation_learning/boundary_segmentation_main.py \
  --server dgx1-1 \
  --config config/boundary_segmentation.yml
