import torch
import kornia
from torch.utils.data import DataLoader

import endoscopy
from utils.io import recursive_scan2df
from datasets import DecordDataset


if __name__ == "__main__":

    prefix = "/media/martin/Samsung_T5/data/endoscopic_data/cholec80/videos"
    data_df = recursive_scan2df(prefix, ".mp4")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    detector = endoscopy.BoundingCircleDetector(device=device)

    ds = DecordDataset(
        data_df=data_df,
        prefix=prefix,
        seq_len=100,
        frame_increment=1,
        frames_between_clips=100,
        num_threads=1
    )
    dl = DataLoader(ds, batch_size=1, num_workers=0)
    log_df = pd.DataFrame(columns=["vid", "frame", "center", "radius"])

    for batch in dl:
        imgs, frame_idcs, vid_idcs = batch
        imgs = imgs[0].to(device).float().permute(0, 3, 1, 2)/255.
        center, radius = detector(imgs, reduction="max")
        # box = endoscopy.max_rectangle_in_circle(imgs.shape, center, radius)
        # crops = kornia.geometry.crop_and_resize(imgs, box, [240, 320])

        seq_data = {
            "vid": vid_idcs[0].numpy().tolist(),
            "frame": frame_idcs[0].numpy().tolist(),
            "center": center.cpu().tolist()*len(frame_idcs[0]),
            "radius": radius.cpu().tolist()*len(frame_idcs[0])
        }

        seq_df = pd.DataFrame(seq_data)
        log_df = log_df.append(
            seq_df, ignore_index=True
        )

        print(log_df)
