import os
import pytorch_lightning as pl
from typing import List

import lightning_modules


class FeatureLSTMModule(pl.LightningModule):
    def __init__(self, shape: List[int], backbone: dict, backbone_prefix: str, lr: float=1e-4, betas: List[float]=[0.9, 0.999], log_n_steps: int=1000):
        super().__init__()
        
        # load trained homography regression backbone
        self.homography_regression = getattr(lightning_modules, backbone['lightning_module']).load_from_checkpoint(
            checkpoint_path=os.path.join(backbone_prefix, backbone['path'], backbone['checkpoint']),
            **backbone['model']
        )

        print(self.backbone)

    def forward(self):
        pass

    def training_step(self):
        pass

    def validation_step(self):
        pass

    def test_step(self):
        # build a test set first
        pass
