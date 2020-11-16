import pytorch_lightning as pl

class SequenceDataModule(pl.LightningDataModule):
    def __init__(self):
        pass

    def setup(self, stage):
        pass

    def transfer_batch_to_device(self, batch, device):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
