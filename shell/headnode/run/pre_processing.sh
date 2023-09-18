#!/bin/bash

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate torch110

python /workspace/homography_imitation_learning/main_pre_processing.py \
  --server headnode \
  --backbone_path ae_cai/resnet/48/25/34/version_0 \
  --data_prefix cholec80_single_video_frames_cropped \
  --in_pkl log.pkl \
  --out_pkl 23_03_07_deep_pre_processed_frame_increment_5_frames_between_clips_1_log.pkl \
  --num_workers 10 \
  --batch_size 400 \
  --frame_increment 5 \
  --frames_between_clips 1
