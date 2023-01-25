#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate torch110

python /workspace/homography_imitation_learning/boundary_segmentation_main.py \
  --server headnode \
  --config config/boundary_segmentation.yml
