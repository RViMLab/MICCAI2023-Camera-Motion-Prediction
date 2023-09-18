import argparse
import os

import cv2
from kornia import tensor_to_image
from kornia.geometry import resize
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader

import lightning_data_modules
import lightning_modules
import lightning_callbacks
from utils.io import load_yaml, natural_keys, scan2df, generate_path
from utils.processing import frame_pairs
from utils.viz import create_blend_from_four_point_homography


def eval(
    camera_motion_predictor: pl.LightningModule,
    camera_motion_estimator_checkpoint: str,
    camera_motion_estimator_config: dict,
    camera_motion_estimator_version: str,
    data_module: pl.LightningDataModule,
    logging_location: str,
    configs: dict,
    output_path: str = "miccai/eval",
) -> None:

    # ckpts = sorted(list(df["file"]), key=natural_keys)
    # homography_regression_ckpt = ckpts[-1]

    device = "cpu"
    if configs["trainer"]["accelerator"] == "gpu":
        device = "cuda"

    callbacks = []
    callbacks.append(
        getattr(lightning_callbacks, "HomographyRegressionCallback")(
            package="lightning_modules",
            module=camera_motion_estimator_config["lightning_module"],
            device=device,
            checkpoint_path=os.path.join(
                logging_location,
                camera_motion_estimator_config["experiment"],
                camera_motion_estimator_version,
                "checkpoints",
                camera_motion_estimator_checkpoint,
            ),
            **camera_motion_estimator_config["model"],
        )
    )

    # add a logger
    logger = TensorBoardLogger(
        save_dir=logging_location,
        name=os.path.join(output_path, "/".join(configs["experiment"].split("/")[-2:])),
    )

    # generate output path
    generate_path(logger.log_dir)

    trainer = pl.Trainer(**configs["trainer"], logger=logger, callbacks=callbacks)
    trainer.test(camera_motion_predictor, datamodule=data_module)
    # print("hello")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/servers.yml", help="Path to config file."
    )
    parser.add_argument("--server", type=str, default="local", help="Server to use.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="miccai/final/cholec80/resnet34/version_2",
        help="Path to experiment, relative to server.logging.location.",
    )
    parser.add_argument(
        "--camera_motion_estimator_version",
        type=str,
        default="version_0",
        help="Version of camera motion estimator.",
    )
    parser.add_argument(
        "--pkl_path",
        type=str,
        default="heichole_single_frames_cropped",
        help="Path to pkl file.",
    )
    parser.add_argument(
        "--pkl_name",
        type=str,
        default="23_03_07_motion_label_window_1_frame_increment_5_frames_between_clips_1_log_test_train.pkl",
        help="Name of pkl file.",
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
        **camera_motion_predictor_config["model"],
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

    # load data module
    camera_motion_predictor_config["data"]["pkl_path"] = args.pkl_path
    camera_motion_predictor_config["data"]["pkl_name"] = args.pkl_name

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
    kwargs["random_frame_offset"] = False
    
    dm = lightning_data_modules.ImageSequenceMotionLabelDataModule(**kwargs)
    dm.setup(stage="test")

    eval(
        camera_motion_predictor=camera_motion_predictor,
        camera_motion_estimator_checkpoint=best_checkpoint,
        camera_motion_estimator_config=camera_motion_estimator_config,
        camera_motion_estimator_version=args.camera_motion_estimator_version,
        data_module=dm,
        logging_location=logging_location,
        configs=camera_motion_predictor_config,
    )


if __name__ == "__main__":
    main()
