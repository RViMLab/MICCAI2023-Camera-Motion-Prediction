runai submit f-lstm-phantom \
  -p mhuber \
  -i aicregistry:5000/mhuber:torch113 \
  -v /nfs/home/mhuber/proj/homography_imitation_learning/:/workspace/homography_imitation_learning \
  -v /nfs/home/mhuber/data:/nfs/home/mhuber/data \
  -v /nfs/home/mhuber/tresorit/homography_imitation_learning_logs:/nfs/home/mhuber/tresorit/homography_imitation_learning_logs \
  -v /nfs/home/mhuber/.torch:/nfs/home/mhuber/.torch \
  -g 1 \
  --environment TORCH_HOME=/nfs/home/mhuber/.torch \
  --large-shm \
  --working-dir /workspace/homography_imitation_learning/ \
  --backoff-limit 1 \
  --run-as-user \
  -- /workspace/homography_imitation_learning/shell/headnode/run_feature_lstm.sh
