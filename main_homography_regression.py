import argparse
import os

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import lightning_data_modules
from lightning_callbacks import RhoCallback
from lightning_modules import homography_regression
from utils.io import generate_path, load_yaml, save_yaml


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
    parser.add_argument("-f", "--find_lr", action="store_true")
    args = parser.parse_args()

    servers = load_yaml(args.servers_file)
    server = servers[args.server]

    configs = load_yaml(args.config)

    # prepare data
    prefix = os.path.join(server["database"]["location"], configs["data"]["pkl_path"])
    df = pd.read_pickle(os.path.join(prefix, configs["data"]["pkl_name"]))

    # load specific data module
    kwargs = {
        "df": df,
        "prefix": prefix,
        "train_split": configs["data"]["train_split"],
        "random_state": configs["data"]["random_state"],
        "batch_size": configs["data"]["batch_size"],
        "num_workers": configs["data"]["num_workers"],
        "rho": configs["data"]["rho"],
        "crp_shape": configs["data"]["crp_shape"],
        "p0": configs["data"]["p0"],
        "seq_len": configs["data"]["seq_len"],
        "unsupervised": configs["data"]["unsupervised"],
        "train_transforms": configs["data"]["train_transforms"],
        "val_transforms": configs["data"]["val_transforms"],
    }

    dm = getattr(lightning_data_modules, configs["lightning_data_module"])(**kwargs)

    # load specific module
    kwargs = configs["model"]

    module = getattr(homography_regression, configs["lightning_module"])(**kwargs)

    logger = TensorBoardLogger(
        save_dir=server["logging"]["location"], name=configs["experiment"]
    )

    # save configs
    generate_path(logger.log_dir)
    save_yaml(os.path.join(logger.log_dir, "config.yml"), configs)

    # callback for homography augmentation edge deviation change
    callbacks = [ModelCheckpoint(**configs["model_checkpoint"])]

    if configs["callback"] is not None:
        callbacks.append(
            [RhoCallback(configs["callback"]["rhos"], configs["callback"]["epochs"])]
        )

    trainer = pl.Trainer(
        **configs["trainer"],
        logger=logger,
        callbacks=callbacks,
    )

    # find learning rate
    if args.find_lr:
        print("Finding learning rate...")
        lr_finder = trainer.tuner.lr_find(module, dm, max_lr=10)
        fig = lr_finder.plot(suggest=True)
        fig.set_dpi(300)

        logger.experiment.add_figure(
            "init/learning_rate_finder", fig, trainer.global_step
        )
        print("Done.")
    else:
        # fit and validation
        trainer.fit(module, dm)

        # test
        trainer.test(datamodule=dm)


if __name__ == "__main__":
    main()
