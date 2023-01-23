import argparse
import os

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import lightning_callbacks
import lightning_data_modules
import lightning_modules
from utils.io import generate_path, load_yaml, natural_keys, save_yaml, scan2df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sf",
        "--servers_file",
        type=str,
        default="config/servers.yml",
        help="Servers file.",
    )
    parser.add_argument(
        "-s", "--server", type=str, default="local", help="Specify server."
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to configuration file."
    )
    parser.add_argument(
        "-bp",
        "--backbone_path",
        type=str,
        help="Path to log folders, relative to server logging location.",
    )
    args = parser.parse_args()

    servers = load_yaml(args.servers_file)
    server = servers[args.server]

    config_path = server["config"]["location"]

    configs = load_yaml(args.config)

    # append configs by backbone
    if args.backbone_path:
        backbone_configs = load_yaml(
            os.path.join(
                server["logging"]["location"], args.backbone_path, "config.yml"
            )
        )
        df = scan2df(
            os.path.join(
                server["logging"]["location"], args.backbone_path, "checkpoints"
            ),
            ".ckpt",
        )
        ckpts = sorted(list(df["file"]), key=natural_keys)
        configs["model"]["homography_regression"] = {
            "lightning_module": backbone_configs["lightning_module"],
            "model": backbone_configs["model"],
            "path": args.backbone_path,
            "checkpoint": "checkpoints/{}".format(ckpts[-1]),
            "experiment": backbone_configs["experiment"],
        }

    # prepare data
    prefix = os.path.join(server["database"]["location"], configs["data"]["pkl_path"])
    df = pd.read_pickle(os.path.join(prefix, configs["data"]["pkl_name"]))

    # load specific data module
    kwargs = {
        "df": df,
        "prefix": prefix,
        **configs["data"]["kwargs"],
    }

    dm = getattr(lightning_data_modules, configs["lightning_data_module"])(**kwargs)

    # load specific module
    kwargs = {
        k: v for k, v in configs["model"].items() if k not in "homography_regression"
    }  # exclude homography regression from kwargs

    module = getattr(lightning_modules, configs["lightning_module"])(**kwargs)

    # inject homography regression into module
    if args.backbone_path:
        configs["model"]["homography_regression"]["model"][
            "pretrained"
        ] = False  # set to false, as loaded anyways
        module.inject_homography_regression(
            homography_regression=configs["model"]["homography_regression"],
            homography_regression_prefix=server["logging"]["location"],
        )

    logger = TensorBoardLogger(
        save_dir=server["logging"]["location"], name=configs["experiment"]
    )

    # save configs
    generate_path(logger.log_dir)
    save_yaml(os.path.join(logger.log_dir, "config.yml"), configs)

    # callbacks
    callbacks = [ModelCheckpoint(**configs["model_checkpoint"])]
    for callback in configs["callbacks"]:
        callbacks.append(
            getattr(lightning_callbacks, callback["name"])(**callback["kwargs"])
        )

    trainer = pl.Trainer(
        **configs["trainer"],
        logger=logger,
        callbacks=callbacks,
    )

    # fit and validation
    trainer.fit(module, dm)

    # # test
    # trainer.test(datamodule=dm)
