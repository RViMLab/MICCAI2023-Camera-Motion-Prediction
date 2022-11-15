import importlib
import argparse
from utils import load_yaml, generate_path, save_yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sf", "--servers_file", type=str, default="config/servers.yml", help="Servers file.")
    parser.add_argument("-s", "--server", type=str, default="local", help="Specify server.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to configuration file.")
    args = parser.parse_args()

    # config
    servers = load_yaml(args.servers_file)
    server = servers[args.server]
    config_path = server["config"]["location"]
    configs = load_yaml(f"{config_path}/{args.config}")

    model = getattr(importlib.import_module(configs["model"]["module"]), configs["model"]["name"])(**configs["model"]["kwargs"])
    data_module = getattr(importlib.import_module(configs["data"]["module"]), configs["data"]["name"])(**configs["data"]["kwargs"])

    # output
    logger = TensorBoardLogger(
        save_dir=server["logging"]["location"],
        name=configs["experiment"]
    )

    # save configs
    generate_path(logger.log_dir)
    save_yaml(f"{logger.log_dir}/config.yml", configs)

    # train
    trainer = pl.Trainer(**configs["trainer"], logger=logger)
    trainer.fit(model, data_module)
    trainer.test(model, data_module)

    # export
