runai submit hil-unsupervised \
  -p mhuber \
  -i 10.202.67.201:32581/mhuber:hil_03 \
  -v /nfs/home/mhuber/proj/homography_imitation_learning/:/workspace/homography_imitation_learning \
  -v /nfs/home/mhuber/data:/nfs/home/mhuber/data \
  -v /nfs/home/mhuber/tresorit/homography_imitation_learning_logs:/nfs/home/mhuber/tresorit/homography_imitation_learning_logs \
  -g 1 \
  --host-ipc \
  --command /workspace/homography_imitation_learning/shell/dgx1-1/run_unsupervised_deep_homography_estimation.sh \
  --working-dir /workspace/homography_imitation_learning/ \
  --run-as-user # defaults to root -> logs will be safed as root too
