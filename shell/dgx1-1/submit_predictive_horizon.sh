runai submit hil-predictive \
  -p mhuber \
  -i 10.202.67.201:32581/mhuber:hil_05 \
  -v /nfs/home/mhuber/proj/homography_imitation_learning/:/workspace/homography_imitation_learning \
  -v /nfs/home/mhuber/data:/nfs/home/mhuber/data \
  -v /nfs/home/mhuber/tresorit/homography_imitation_learning_logs:/nfs/home/mhuber/tresorit/homography_imitation_learning_logs \
  -v /nfs/home/mhuber/.torch:/nfs/home/mhuber/.torch \
  -g 1 \
  --environment TORCH_HOME=/nfs/home/mhuber/.torch \
  --host-ipc \
  --command /workspace/homography_imitation_learning/shell/dgx1-1/run_predictive_horizon.sh \
  --working-dir /workspace/homography_imitation_learning/ \
  --run-as-user # defaults to root -> logs will be saved as root too
