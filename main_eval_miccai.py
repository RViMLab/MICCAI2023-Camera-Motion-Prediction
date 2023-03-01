import argparse
import os

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import lightning_data_modules
import lightning_modules
from utils.io import load_yaml, natural_keys, scan2df


def test(module: pl.LightningModule, test_dataloader: DataLoader) -> None:
    for batch in test_dataloader:
        img, tf_img, frame_idcs, vid_idcs = batch
        print(img.shape)


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
    args = parser.parse_args()
    server = load_yaml(args.config)[args.server]

    database_location = server["database"]["location"]
    logging_location = server["logging"]["location"]

    # search checkpoints
    df = scan2df(
        os.path.join(server["logging"]["location"], args.experiment, "checkpoints"),
        ".ckpt",
    )
    best_checkpoint = sorted(list(df["file"]), key=natural_keys)[-1]

    # load best checkpoint
    config = load_yaml(os.path.join(logging_location, args.experiment, "config.yml"))
    model = getattr(lightning_modules, config["lightning_module"]).load_from_checkpoint(
        checkpoint_path=os.path.join(
            server["logging"]["location"],
            args.experiment,
            "checkpoints",
            best_checkpoint,
        ),
        **config["model"]
    )
    model.freeze()
    model = model.eval()

    # load data module
    df = pd.read_pickle(
        os.path.join(
            database_location, config["data"]["pkl_path"], config["data"]["pkl_name"]
        )
    )
    kwargs = {
        "df": df,
        "prefix": os.path.join(database_location, config["data"]["pkl_path"]),
        **config["data"]["kwargs"],
    }

    dm = getattr(lightning_data_modules, config["lightning_data_module"])(**kwargs)
    dm.setup(stage="test")
    test_dataloader = dm.test_dataloader()

    # run tests
    test(module=model, test_dataloader=test_dataloader)


if __name__ == "__main__":
    main()
