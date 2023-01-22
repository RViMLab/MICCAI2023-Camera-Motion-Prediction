import os

import importlib
import pytorch_lightning as pl
import torch

import lightning_modules
from utils.processing import frame_pairs


class HomographyPredictorModule(pl.LightningModule):
    def __init__(
        self,
        predictor: dict,
        optimizer: dict,
        loss: dict,
    ):
        super().__init__()
        self._predictor = getattr(
            importlib.import_module(predictor["module"], predictor["name"])
        )(**predictor["kwargs"])

        self._optimizer = getattr(
            importlib.import_module(optimizer["module"], optimizer["name"])
        )(**optimizer["kwargs"])

        self._loss = getattr(
            importlib.import_module(loss["module"], loss["name"])
        )(**loss["kwargs"])

        self._homography_regression = None

    def inject_homography_regression(
        self, homography_regression: dict, homography_regression_prefix: str
    ):
        # load trained homography regression model
        self._homography_regression = getattr(
            lightning_modules, homography_regression["lightning_module"]
        ).load_from_checkpoint(
            checkpoint_path=os.path.join(
                homography_regression_prefix,
                homography_regression["path"],
                homography_regression["checkpoint"],
            ),
            **homography_regression["model"]
        )
        self._homography_regression = self._homography_regression.eval()
        self._homography_regression.freeze()

    def on_train_epoch_start(self):
        if self._homography_regression:
            self._homography_regression = self._homography_regression.eval()
            self._homography_regression.freeze()

    def configure_optimizers(self):
        return self._optimizer

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        pass

    def training_step(self, batch, batch_idx):
        (
            tf_imgs,
            duvs_reg,
            frame_idcs,
            vid_idcs,
        ) = batch  # transformed images and four point homography
        tf_imgs = tf_imgs.float() / 255.0
        duvs_reg = duvs_reg.float()

        duvs_pred = self(tf_imgs)
        
                

    def validation_step(self, batch, batch_idx):
        (
            tf_imgs,
            duvs_reg,
            frame_idcs,
            vid_idcs,
        ) = batch  # transformed images and four point homography
        tf_imgs = tf_imgs.float() / 255.0
        duvs_reg = duvs_reg.float()



    def test_step(self, batch, batch_idx):
        pass
