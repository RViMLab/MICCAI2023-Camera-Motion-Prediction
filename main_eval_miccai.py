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


def visualize(
    camera_motion_predictor: pl.LightningModule,
    camera_motion_estimator: pl.LightningModule,
    test_dataloader: DataLoader,
    output_path: str,
    preview_horizon: int = 1,
) -> None:

    # run prediction vs estimation on an entire sequence
    cnt = 0
    max_cnt = 200
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

        # plot estimated vs predicted camera motion
        blends = create_blend_from_four_point_homography(
            preview_imgs_i, preview_imgs_ip1, duvs_esti
        )

        blends = resize(blends, [480, 640])
        blends = tensor_to_image(blends[0], keepdim=False)
        blends = (blends * 255.0).astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, f"esti_blend_{cnt}.png"), blends)

        blends = create_blend_from_four_point_homography(
            preview_imgs_i, preview_imgs_ip1, duvs_pred
        )

        blends = resize(blends, [480, 640])
        blends = tensor_to_image(blends[0], keepdim=False)
        blends = (blends * 255.0).astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, f"pred_blend_{cnt}.png"), blends)

        preview_imgs_i = resize(preview_imgs_i, [480, 640])
        preview_imgs_i = tensor_to_image(preview_imgs_i[0], keepdim=False)
        preview_imgs_i = (preview_imgs_i * 255.0).astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, f"img_{cnt}.png"), preview_imgs_i)

        # break
        cnt += 1
        print(frame_idcs[:, -2])
        print(f"{cnt}/{max_cnt}")
        if cnt >= max_cnt:
            break


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
                camera_motion_estimator_config["ae_cai/resnet/48/25/34"],
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

    # trainer = pl.Trainer(**configs["experiment"], logger=logger, callbacks=callbacks)
    # trainer.test(camera_motion_predictor, datamodule=data_module)
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
        "--visualize", action="store_true", help="Visualize camera motion prediction."
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

    # # load camera motion estimator with best checkpoint
    # camera_motion_estimator = getattr(
    #     lightning_modules, camera_motion_estimator_config["lightning_module"]
    # ).load_from_checkpoint(
    #     checkpoint_path=os.path.join(
    #         logging_location,
    #         camera_motion_estimator_config["experiment"],
    #         args.camera_motion_estimator_version,
    #         "checkpoints",
    #         best_checkpoint,
    #     ),
    #     **camera_motion_estimator_config["model"]
    # )

    # camera_motion_estimator.to(device)
    # camera_motion_estimator.freeze()
    # camera_motion_estimator = camera_motion_estimator.eval()

    # load data module
    if args.pkl_path and args.pkl_name:
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
    kwargs["batch_size"] = 1
    kwargs["random_frame_offset"] = False
    assert kwargs["frames_between_clips"] == kwargs["frame_increment"]

    dm = lightning_data_modules.ImageSequenceDataModule(**kwargs)
    dm.setup(stage="test")
    test_dataloader = dm.test_dataloader()

    # # run tests
    # if args.visualize:
    #     visualize(
    #         camera_motion_predictor=camera_motion_predictor,
    #         camera_motion_estimator=camera_motion_estimator,
    #         test_dataloader=test_dataloader,
    #     )

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
