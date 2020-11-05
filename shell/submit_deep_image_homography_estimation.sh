runai submit hil-deep-image-homography-estimation \
  -p mhuber \
  -i 10.202.67.201:32581/mhuber:hi \
  -v /nfs/home/mhuber/proj/homography_imitation_learning/:/workspace/homography_imitation_learning \
  -v /nfs/home/mhuber/data:/nfs/home/mhuber/data \
  -v /nfs/home/mhuber/logs:/nfs/home/mhuber/logs \
  -g 1 \
  --host-ipc \
  --command /workspace/homography_imitation_learning/shell/run_deep_image_homography_estimation.sh \
  --working-dir /workspace/homography_imitation_learning/ \
  --run-as-user # defaults to root -> logs will be safed as root too
