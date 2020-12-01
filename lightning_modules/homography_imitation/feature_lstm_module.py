import pytorch_lightning as pl

import lightning_modules


class FeatureLSTMModule(pl.LightningModule):
    def __init__(self):
        # self.homography_regression = getattr(lightning_modules, ).load_from_checkpoint(kwargs)
        pass

    def forward(self):
        pass

    def training_step(self):
        pass

    def validation_step(self):
        pass

    def test_step(self):
        # build a test set first
        pass
