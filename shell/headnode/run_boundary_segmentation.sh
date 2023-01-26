#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate torch110

python /workspace/homography_imitation_learning/main_boundary_segmentation.py \
  --server headnode \
  --config config/boundary_segmentation.yml
