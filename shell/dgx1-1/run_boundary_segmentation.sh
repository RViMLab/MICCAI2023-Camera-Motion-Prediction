#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate torch110

python /workspace/homography_imitation_learning/boundary_segmentation_main.py \
  --server dgx1-1 \
  --config config/boundary_segmentation.yml
