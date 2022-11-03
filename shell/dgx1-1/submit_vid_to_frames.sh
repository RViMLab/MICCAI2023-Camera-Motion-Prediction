runai submit vid2frames \
  -p mhuber \
  -i 10.202.67.207:5000/mhuber:torch110 \
  -v /nfs/home/mhuber/proj/homography_imitation_learning/:/workspace/homography_imitation_learning \
  -v /nfs/home/mhuber/data:/nfs/home/mhuber/data \
  -v /nfs/home/mhuber/tresorit/homography_imitation_learning_logs:/nfs/home/mhuber/tresorit/homography_imitation_learning_logs \
  -v /nfs/home/mhuber/.torch:/nfs/home/mhuber/.torch \
  -g 0.2 \
  --cpu-limit 10 \
  --environment TORCH_HOME=/nfs/home/mhuber/.torch \
  --working-dir /workspace/homography_imitation_learning/ \
  --run-as-user \
  --command -- /workspace/homography_imitation_learning/shell/dgx1-1/run_vid_to_frames.sh
