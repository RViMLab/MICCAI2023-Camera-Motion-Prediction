import os
import argparse
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from utils.io import load_yaml
from lightning_modules import homography_regression
import lightning_data_modules


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-sf', '--servers_file', type=str, default='configs/servers.yml', help='Servers file.')
    parser.add_argument('-s', '--server', type=str, default='local', help='Specify server.')
    parser.add_argument('-c', '--configs', type=str, required=True, help='Path to configuration file.')
    args = parser.parse_args()

    servers = load_yaml(args.servers_file)
    server = servers[args.server]

    configs = load_yaml(args.configs)

    # prepare data
    prefix = os.path.join(server['database']['location'], configs['data']['pkl_path'])
    df = pd.read_pickle(os.path.join(
            prefix,
            configs['data']['pkl_name']
    ))

    # load specific data module
    kwargs = {
        'df': df, 
        'prefix': prefix, 
        'train_split': configs['data']['train_split'],
        'random_state': configs['data']['random_state'],
        'batch_size': configs['data']['batch_size'],
        'num_workers': configs['data']['num_workers'],
        'rho': configs['data']['rho'],
        'crp_shape': configs['data']['crp_shape'],
        'unsupervised': configs['data']['unsupervised']
    } 

    dm = getattr(lightning_data_modules, configs['lightning_data_module'])(**kwargs)
    dm.setup()

    # load specific module
    shape = next(iter(dm.train_dataloader()))['img_seq_crp'][0].shape
    kwargs = {
        'shape': shape[-3:],
        'lr': configs['hparams']['lr'],
        'betas': configs['hparams']['betas']
    }

    module = getattr(homography_regression, configs['lightning_module'])(**kwargs)

    logger = TensorBoardLogger(
        save_dir=server['logging']['location'],
        name=configs['experiment']
    )

    trainer = pl.Trainer(
        max_epochs=configs['trainer']['max_epochs'],
        logger=logger,
        log_every_n_steps=configs['trainer']['log_every_n_steps'],
        limit_train_batches=configs['trainer']['limit_train_batches'],
        limit_val_batches=configs['trainer']['limit_val_batches'],
        limit_test_batches=configs['trainer']['limit_test_batches'],
        gpus=configs['trainer']['gpus']
    )

    # fit and validation
    trainer.fit(module, dm)

    # # test
    trainer.test()
