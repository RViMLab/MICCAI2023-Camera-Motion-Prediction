import os
import argparse
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from utils.io import load_yaml, save_yaml, load_pickle, save_pickle, generate_path, scan2df, natural_keys
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

    config_path = server['configs']['location']

    configs = load_yaml(args.configs)

    # append configs by backbone
    backbone_configs = load_yaml(os.path.join(server['logging']['location'], args.backbone_path, 'configs.yml'))
    df = scan2df(os.path.join(server['logging']['location'], args.backbone_path, 'checkpoints'), '.ckpt')
    ckpts = sorted(list(df['file']), key=natural_keys)
    configs['model']['homography_regression'] = {
        'lightning_module': backbone_configs['lightning_module'],
        'model': backbone_configs['model'],
        'path': args.backbone_path,
        'checkpoint': 'checkpoints/{}'.format(ckpts[-1]),
        'experiment': backbone_configs['experiment']
    }

    # prepare data
    prefix = os.path.join(server['database']['location'])
    meta_df = pd.read_pickle(os.path.join(config_path, configs['data']['meta_df']))

    # load video meta data if existing, returns None if none existent
    train_md = load_pickle(os.path.join(server['configs']['location'], configs['data']['train_metadata']))
    val_md = load_pickle(os.path.join(server['configs']['location'], configs['data']['val_metadata']))
    test_md = load_pickle(os.path.join(server['configs']['location'], configs['data']['test_metadata']))

    # load specific data module
    kwargs = {
        'meta_df': meta_df,
        'prefix': prefix,
        'clip_length_in_frames': configs['data']['clip_length_in_frames'],
        'frames_between_clips': configs['data']['frames_between_clips'],
        'frame_stride': configs['data']['frame_stride'],
        'train_split': configs['data']['train_split'],
        'batch_size': configs['data']['batch_size'],
        'num_workers': configs['data']['num_workers'],
        'random_state': configs['data']['random_state'],
        'train_metadata': train_md,
        'val_metadata': val_md,
        'test_metadata': test_md
    }

    dm = getattr(lightning_data_modules, configs['lightning_data_module'])(**kwargs)
    dm.prepare_data()

    train_md, val_md, test_md = dm.setup()

    save_pickle(os.path.join(server['configs']['location'], configs['data']['train_metadata']), train_md)
    save_pickle(os.path.join(server['configs']['location'], configs['data']['val_metadata']), val_md)
    save_pickle(os.path.join(server['configs']['location'], configs['data']['test_metadata']), test_md)

    # load specific module
    kwargs = configs['model']

    module = getattr(lightning_modules, configs['lightning_module'])(**kwargs, homography_regression_prefix=server['logging']['location'])

    logger = TensorBoardLogger(
        save_dir=server['logging']['location'],
        name=configs['experiment']
    )

    # save configs
    generate_path(logger.log_dir)
    save_yaml(os.path.join(logger.log_dir, 'configs.yml'), configs)
    meta_df.to_pickle(os.path.join(logger.log_dir, configs['data']['meta_df']))

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
