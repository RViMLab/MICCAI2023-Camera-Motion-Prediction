import os
import pytorch_lightning as pl
import pandas as pd
import argparse

from utils.io import load_yaml
import lightning_data_modules
import lightning_modules

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
        'train_transforms': configs['data']['train_transforms'],
        'val_transforms': configs['data']['val_transforms']
    }

    dm = getattr(lightning_data_modules, configs['lightning_data_module'])

    # load specific module
    kwargs = configs['model']

    module = getattr(lightning_modules, configs['lightning_module'])

    trainer = pl.Trainer(

    )

    trainer.fit(module, dm)

    # trainer.test()
