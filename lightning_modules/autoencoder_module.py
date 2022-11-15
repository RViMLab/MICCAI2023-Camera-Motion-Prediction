import torch
import importlib
from typing import Any, Optional, Union, List
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT


class UNetModule(pl.LightningModule):
    def __init__(self, segmentation_model: dict, loss_function: dict, optimizer: dict, scheduler: dict=None) -> None:
        super().__init__()
        self._model = getattr(
            importlib.import_module(segmentation_model["module"]),
            segmentation_model["name"]
        )(**segmentation_model["kwargs"])
        
        self._loss_function = getattr(
            importlib.import_module(loss_function["module"]),
            loss_function["name"]
        )(**loss_function["kwargs"])

        self._optimizer = getattr(
            importlib.import_module(optimizer["module"]),
            optimizer["name"]
        )(params=self._model.parameters(), **optimizer["kwargs"])

        self._scheduler = None
        if scheduler:
            self._scheduler = getattr(
                importlib.import_module(scheduler["module"]),
                scheduler["name"]
            )(optimizer=self._optimizer, **scheduler["kwargs"])

        # single log
        self._val_logged = False

    def forward(self, x) -> Any:
        return self._model(x)

    def configure_optimizers(self) -> Any:
        if self._scheduler:
            return [self._optimizer], [self._scheduler]
        return self._optimizer

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        target = batch
        target = target.float()/255.
        output = self(target)
        loss = self._loss_function(output, target)
        self.log("train/loss", loss)
        return {
            "loss": loss
        }

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        target = batch
        target = target.float()/255.
        output = self(target)
        loss = self._loss_function(output, target)
        self.log("val/loss", loss)

        if not self._val_logged:
            self.logger.experiment.add_images('val/target', target, self.global_step)
            self.logger.experiment.add_images('val/output', output, self.global_step)
            self._val_logged = True

        return {
            "loss": loss
        }

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self._val_logged = False
        return super().validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        target = batch
        target = target.float()/255.
        output = self(target)
        loss = self._loss_function(output, target)
        self.log("test/loss", loss)
        return {
            "loss": loss
        }
