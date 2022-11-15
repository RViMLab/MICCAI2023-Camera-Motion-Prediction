import importlib
from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT


class UNetModule(pl.LightningModule):
    def __init__(self, segmentation_model: dict, loss_function: dict) -> None:
        super().__init__()
        self._model = getattr(
            importlib.import_module(segmentation_model["module"]),
            segmentation_model["name"]
        )(**segmentation_model["kwargs"])
        
        self._loss_function = getattr(
            importlib.import_module(loss_function["module"]),
            loss_function["name"]
        )(**loss_function["kwargs"])

    def forward(self, x) -> Any:
        return self._model(x)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x = batch
        y = self._model(x)
        loss = self._loss_function(y, x)
        self.log("train/loss", loss)
        return {
            "loss": loss
        }

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        x = batch
        y = self._model(x)
        loss = self._loss_function(y, x)
        self.log("val/loss", loss)
        return {
            "loss": loss
        }

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        x = batch
        y = self._model(x)
        loss = self._loss_function(y, x)
        self.log("test/loss", loss)
        return {
            "loss": loss
        }
