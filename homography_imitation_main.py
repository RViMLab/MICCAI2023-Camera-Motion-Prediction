import os
import argparse
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from utils.io import load_yaml, save_yaml, generate_path, scan2df, natural_keys
import lightning_data_modules
import lightning_modules


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-sf', '--servers_file', type=str, default='configs/servers.yml', help='Servers file.')
    parser.add_argument('-s', '--server', type=str, default='local', help='Specify server.')
    parser.add_argument('-c', '--configs', type=str, required=True, help='Path to configuration file.')
    parser.add_argument('-bp', '--backbone_path', type=str, required=True, help='Path to log folders, relative to server logging location.')
    args = parser.parse_args()

    servers = load_yaml(args.servers_file)
    server = servers[args.server]

    configs = load_yaml(args.configs)

    # append configs by backbone
    backbone_configs = load_yaml(os.path.join(server['logging']['location'], args.backbone_path, 'configs.yml'))
    df = scan2df(os.path.join(server['logging']['location'], args.backbone_path, 'checkpoints'), '.ckpt')
    ckpts = sorted(list(df['files']), key=natural_keys)
    configs['model']['backbone'] = {
        'lightning_module': backbone_configs['lightning_module'],
        'model': backbone_configs['model'],
        'path': args.backbone_path,
        'checkpoint': 'checkpoints/{}'.format(ckpts[-1]),
        'experiment': backbone_configs['experiment']
    }

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
        'train_transforms': configs['data']['train_transforms'],
        'val_transforms': configs['data']['val_transforms']
    }

    dm = getattr(lightning_data_modules, configs['lightning_data_module'])

    # load specific module
    kwargs = configs['model']

    module = getattr(lightning_modules, configs['lightning_module'])

    logger = TensorBoardLogger(
        save_dir=server['logging']['location'],
        name=configs['experiment']
    )

    # save configs
    generate_path(logger.log_dir)
    save_yaml(os.path.join(logger.log_dir, 'configs.yml'), configs)

    trainer = pl.Trainer(
        max_epochs=configs['trainer']['max_epochs'],
        logger=logger,
        log_every_n_steps=configs['trainer']['log_every_n_steps'],
        limit_train_batches=configs['trainer']['limit_train_batches'],
        limit_val_batches=configs['trainer']['limit_val_batches'],
        limit_test_batches=configs['trainer']['limit_test_batches'],
        gpus=configs['trainer']['gpus']
    )

    trainer.fit(module, dm)

    # trainer.test()
