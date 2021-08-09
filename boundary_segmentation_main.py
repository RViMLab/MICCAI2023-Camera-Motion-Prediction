import os
import argparse
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import lightning_data_modules

from lightning_modules import image_segmentation
import lightning_data_modules
from utils.io import load_yaml, generate_path, save_yaml


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-sf', '--servers_file', type=str, default='config/servers.yml', help='Servers file.')
    parser.add_argument('-s', '--server', type=str, default='local', help='Specify server.')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to configuration file.')
    args = parser.parse_args()

    servers = load_yaml(args.servers_file)
    server = servers[args.server]

    configs = load_yaml(args.config)

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
        'batch_size': configs['data']['batch_size'],
        'num_workers': configs['data']['num_workers'],
        'random_state': configs['data']['random_state'],
        'train_image_transforms': configs['data']['train_image_transforms'],
        'train_spatial_transforms': configs['data']['train_spatial_transforms'],
        'val_image_transforms': configs['data']['val_image_transforms'],
        'val_spatial_transforms': configs['data']['val_spatial_transforms'],
        'test_image_transforms': configs['data']['test_image_transforms'],
        'test_spatial_transforms': configs['data']['test_spatial_transforms']
    }

    dm = getattr(lightning_data_modules, configs['lightning_data_module'])(**kwargs)
    dm.setup()

    # load specific module
    kwargs = configs['model']

    module = getattr(image_segmentation, configs['lightning_module'])(**kwargs)

    logger = TensorBoardLogger(
        save_dir=server['logging']['location'],
        name=configs['experiment']
    )

    # save configs
    generate_path(logger.log_dir)
    save_yaml(os.path.join(logger.log_dir, 'config.yml'), configs)

    # monitor
    monitor_callback = ModelCheckpoint(**configs['model_checkpoint'])

    # create trainer
    trainer = pl.Trainer(
        max_epochs=configs['trainer']['max_epochs'],
        logger=logger,
        log_every_n_steps=configs['trainer']['log_every_n_steps'],
        limit_train_batches=configs['trainer']['limit_train_batches'],
        limit_val_batches=configs['trainer']['limit_val_batches'],
        limit_test_batches=configs['trainer']['limit_test_batches'],
        gpus=configs['trainer']['gpus'],
        fast_dev_run=configs['trainer']['fast_dev_run'],
        profiler=configs['trainer']['profiler'],
        distributed_backend=configs['trainer']['distributed_backend'],
        callbacks=[monitor_callback]
    )

    # fit and validation
    trainer.fit(module, dm)

    # test
    trainer.test()
