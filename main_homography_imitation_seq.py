import argparse
import importlib
import os

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import lightning_callbacks
import lightning_data_modules
import lightning_modules
from utils.io import generate_path, load_yaml, natural_keys, save_yaml, scan2df


def main() -> None:
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
        "-hr",
        "--homography_regression",
        type=str,
        help="Path to log folders, relative to server logging location.",
    )
    args = parser.parse_args()

    servers = load_yaml(args.servers_file)
    server = servers[args.server]

    config_path = server["config"]["location"]

    configs = load_yaml(args.config)

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

    module = getattr(lightning_modules, configs["lightning_module"])(**configs["model"])

    logger = TensorBoardLogger(
        save_dir=server["logging"]["location"], name=configs["experiment"]
    )

    # callbacks
    callbacks = []
    for callback in configs["callbacks"]:
        callbacks.append(
            getattr(importlib.import_module(callback["module"]), callback["name"])(
                **callback["kwargs"]
            )
        )

    # load homography regression callback
    if args.homography_regression:
        homography_regression_config = load_yaml(
            os.path.join(
                server["logging"]["location"], args.homography_regression, "config.yml"
            )
        )
        df = scan2df(
            os.path.join(
                server["logging"]["location"], args.homography_regression, "checkpoints"
            ),
            ".ckpt",
        )
        ckpts = sorted(list(df["file"]), key=natural_keys)
        homography_regression_ckpt = ckpts[-1]

        device = "cpu"
        if configs["trainer"]["accelerator"] == "gpu":
            device = "cuda"

        callbacks.append(
            getattr(lightning_callbacks, "HomographyRegressionCallback")(
                preview_horizon=1,
                package="lightning_modules",
                module=homography_regression_config["lightning_module"],
                device=device,
                checkpoint_path=os.path.join(
                    server["logging"]["location"],
                    args.homography_regression,
                    "checkpoints",
                    homography_regression_ckpt,
                ),
                **homography_regression_config["model"],
            )
        )

    # save configs
    generate_path(logger.log_dir)
    save_yaml(os.path.join(logger.log_dir, "config.yml"), configs)
    save_yaml(
        os.path.join(logger.log_dir, "homography_regression_config.yml"),
        homography_regression_config,
    )

    # trainer
    trainer = pl.Trainer(
        **configs["trainer"],
        logger=logger,
        callbacks=callbacks,
    )

    # fit and validation
    trainer.fit(module, dm)

    # test
    trainer.test(datamodule=dm)


if __name__ == "__main__":
    main()
