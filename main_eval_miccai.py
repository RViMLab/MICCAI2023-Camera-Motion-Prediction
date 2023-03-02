import argparse
import os

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import lightning_data_modules
import lightning_modules
from utils.io import load_yaml, natural_keys, scan2df
from utils.processing import frame_pairs


def test(
    camera_motion_predictor: pl.LightningModule,
    camera_motion_estimator: pl.LightningModule,
    test_dataloader: DataLoader,
    preview_horizon: int = 1,
) -> None:
    for batch in test_dataloader:
        imgs, tf_imgs, frame_idcs, vid_idcs = batch

        # process images
        B, T, C, H, W = imgs.shape
        imgs = imgs.to(camera_motion_predictor.device)
        imgs = imgs.float() / 255.0

        # inference camera motion predictor
        recall_imgs = imgs[:, :-preview_horizon]
        recall_imgs = recall_imgs.view(B, -1, H, W)

        with torch.no_grad():
            duvs_pred = camera_motion_predictor(recall_imgs)

        # inference camera motion estimator
        preview_imgs = imgs[:, -preview_horizon - 1 :]
        preview_imgs_i, preview_imgs_ip1 = frame_pairs(preview_imgs)

        with torch.no_grad():
            preview_imgs_i, preview_imgs_ip1 = preview_imgs_i.view(
                -1, C, H, W
            ), preview_imgs_ip1.view(-1, C, H, W)
            duvs_esti = camera_motion_estimator(preview_imgs_i, preview_imgs_ip1)

        print(duvs_pred.shape)
        print(duvs_esti.shape)
        break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/servers.yml", help="Path to config file."
    )
    parser.add_argument("--server", type=str, default="local", help="Server to use.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="miccai/motion_labels/heichole/resnet50/version_0",
        help="Path to experiment, relative to server.logging.location.",
    )
    parser.add_argument(
        "--camera_motion_predictor",
        type=str,
        default="ae_cai/resnet/48/25/34/version_0",
        help="Path camera motion estimator, relative to server.logging.location.",
    )
    parser.add_argument(
        "--camera_motion_estimator_version",
        type=str,
        default="version_0",
        help="Version of camera motion estimator.",
    )
    args = parser.parse_args()
    server = load_yaml(args.config)[args.server]

    database_location = server["database"]["location"]
    logging_location = server["logging"]["location"]

    # search checkpoints
    df = scan2df(
        os.path.join(logging_location, args.experiment, "checkpoints"),
        ".ckpt",
    )
    best_checkpoint = sorted(list(df["file"]), key=natural_keys)[-1]

    # load camera motion predictor with best checkpoint
    camera_motion_predictor_config = load_yaml(
        os.path.join(logging_location, args.experiment, "config.yml")
    )
    camera_motion_predictor = getattr(
        lightning_modules, camera_motion_predictor_config["lightning_module"]
    ).load_from_checkpoint(
        checkpoint_path=os.path.join(
            logging_location,
            args.experiment,
            "checkpoints",
            best_checkpoint,
        ),
        **camera_motion_predictor_config["model"]
    )

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    camera_motion_predictor.to(device)
    camera_motion_predictor.freeze()
    camera_motion_predictor = camera_motion_predictor.eval()

    # load camera motion estimator configs
    camera_motion_estimator_config = load_yaml(
        os.path.join(
            logging_location, args.experiment, "homography_regression_config.yml"
        )
    )

    # search checkpoints
    df = scan2df(
        os.path.join(
            logging_location,
            camera_motion_estimator_config["experiment"],
            args.camera_motion_estimator_version,
            "checkpoints",
        ),
        ".ckpt",
    )
    best_checkpoint = sorted(list(df["file"]), key=natural_keys)[-1]

    # load camera motion estimator with best checkpoint
    camera_motion_estimator = getattr(
        lightning_modules, camera_motion_estimator_config["lightning_module"]
    ).load_from_checkpoint(
        checkpoint_path=os.path.join(
            logging_location,
            camera_motion_estimator_config["experiment"],
            args.camera_motion_estimator_version,
            "checkpoints",
            best_checkpoint,
        ),
        **camera_motion_estimator_config["model"]
    )

    camera_motion_estimator.to(device)
    camera_motion_estimator.freeze()
    camera_motion_estimator = camera_motion_estimator.eval()

    # load data module
    df = pd.read_pickle(
        os.path.join(
            database_location,
            camera_motion_predictor_config["data"]["pkl_path"],
            camera_motion_predictor_config["data"]["pkl_name"],
        )
    )
    kwargs = {
        "df": df,
        "prefix": os.path.join(
            database_location, camera_motion_predictor_config["data"]["pkl_path"]
        ),
        **camera_motion_predictor_config["data"]["kwargs"],
    }

    dm = lightning_data_modules.ImageSequenceDataModule(**kwargs)
    dm.setup(stage="test")
    test_dataloader = dm.test_dataloader()

    # run tests
    test(
        camera_motion_predictor=camera_motion_predictor,
        camera_motion_estimator=camera_motion_estimator,
        test_dataloader=test_dataloader,
    )


if __name__ == "__main__":
    main()
