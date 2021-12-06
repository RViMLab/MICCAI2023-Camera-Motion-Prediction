#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate torch110

python /workspace/homography_imitation_learning/homography_imitation_main.py \
  --server dgx1-1 \
  --config config/next_view_dgx.yml \
  --backbone_path endoscopy_view/resnet/34/version_0
