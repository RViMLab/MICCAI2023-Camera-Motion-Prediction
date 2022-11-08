#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate torch110

python /workspace/homography_imitation_learning/pre_processing_main.py \
  --server dgx1-1 \
  --backbone_path ae_cai/resnet/48/25/34/version_0 \
  --data_prefix cholec80_single_video_frames_cropped \
  --in_pkl log.pkl \
  --out_pkl pre_processed_5th_frame_log.pkl \
  --num_workers 20 \
  --batch_size 400 \
  --frame_increment 5 \
  --frames_between_clips 5
