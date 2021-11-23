#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda env update -f /workspace/homography_imitation_learning/env_torch110.yml
conda activate torch110

python /workspace/homography_imitation_learning/pre_processing_main.py \
  --server dgx1-1 \
  --backbone_path endoscopy_view/resnet/34/version_0 \
  --data_prefix cholec80_frames \
  --in_pkl log.pkl \
  --out_pkl pre_processed_log.pkl \
  --num_workers 12 \
  --batch_size 400
