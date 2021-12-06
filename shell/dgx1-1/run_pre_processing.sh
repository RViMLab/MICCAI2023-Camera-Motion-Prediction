#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate torch110

python /workspace/homography_imitation_learning/pre_processing_main.py \
  --server dgx1-1 \
  --backbone_path endoscopy_view/resnet/34/version_0 \
  --data_prefix cholec80_frames \
  --in_pkl log.pkl \
  --out_pkl pre_processed_5th_frame_log.pkl \
  --num_workers 20 \
  --batch_size 400 \
  --nth_frame 5
