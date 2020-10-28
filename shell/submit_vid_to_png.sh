runai submit hil-vid-to-png \
  -p mhuber \
  -i 10.202.67.201:32581/mhuber:hil \
  -v /nfs/home/mhuber/proj/homography_imitation_learning/:/workspace/homography_imitation_learning \
  -v /nfs/home/mhuber/data:/nfs/home/mhuber/data \
  -g 0 \
  --command /workspace/homography_imitation_learning/shell/run_vid_to_png.sh \
  --working-dir /workspace/homography_imitation_learning/ \
  --run-as-user # defaults to root -> logs will be safed as root too
