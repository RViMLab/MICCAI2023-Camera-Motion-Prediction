import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from lightning_modules import SupervisedHomographyModule
from lightning_data_modules import ConsecutiveDataModule


if __name__ == '__main__':

    prefix = '/media/martin/Samsung_T5/data/endoscopic_data/camera_motion_separated_png/without_camera_motion'
    pkl_name = 'log_without_camera_motion_seq_len_2.pkl'
    df = pd.read_pickle(os.path.join(prefix, pkl_name))

    cdm = ConsecutiveDataModule(df, prefix, train_split=0.8, batch_size=2, num_workers=2, rho=32, crp_shape=[240, 320])
    cdm.setup()

    shape = next(iter(cdm.train_dataloader()))['img_seq'][0].shape
    shm = SupervisedHomographyModule(
        shape=[2*shape[1], shape[2], shape[3]]
    )

    logger = TensorBoardLogger(save_dir='tb_log', name='experiment1')

    trainer = pl.Trainer(
        max_epochs=10,
        logger=logger,
        limit_train_batches=0.001,
        limit_val_batches=0.001,
        limit_test_batches=0.001,
        gpus=1
    )

    # fit and validation
    trainer.fit(shm, cdm)

    # # test
    trainer.test()
