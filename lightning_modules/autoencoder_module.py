import importlib
from collections import OrderedDict
from typing import Any, List, Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT


class GANAutoencoderModule(pl.LightningModule):

    def __init__(
        self,
        generator_model: dict,
        discriminator_model: dict,
        generator_optimizer: List[dict],
        discriminator_optimizer: List[dict],
        alpha: float=1e-2,
        generator_scheduler: List[dict]=None,
        discriminator_scheduler: List[dict]=None
    ) -> None:
        super().__init__()
        self._generator = getattr(
            importlib.import_module(generator_model["module"]),
            generator_model["name"]
        )(**generator_model["kwargs"])

        self._discriminator = getattr(
            importlib.import_module(discriminator_model["module"]),
            discriminator_model["name"]
        )(**discriminator_model["kwargs"])

        self._mse_loss = torch.nn.MSELoss() # maintain original contents
        self._bce_loss = torch.nn.BCELoss() # make image look realistic

        self._generator_optimizer = getattr(
            importlib.import_module(generator_optimizer["module"]),
            generator_optimizer["name"]
        )(params=self._generator.parameters(), **generator_optimizer["kwargs"])

        self._discriminator_optimizer = getattr(
            importlib.import_module(discriminator_optimizer["module"]),
            discriminator_optimizer["name"]
        )(params=self._discriminator.parameters(), **discriminator_optimizer["kwargs"])

        self._alpha = alpha

        self._generator_scheduler = None
        self._discriminator_scheduler = None

        if generator_scheduler:
            self._generator_scheduler = getattr(
                importlib.import_module(generator_scheduler["module"]),
                generator_scheduler["name"]
            )(optimizer=self._generator_optimizer, **generator_scheduler["kwargs"])

        if discriminator_scheduler:
            self._generator_scheduler = getattr(
                importlib.import_module(discriminator_scheduler["module"]),
                discriminator_scheduler["name"]
            )(optimizer=self._generator_optimizer, **discriminator_scheduler["kwargs"])

        # single log
        self._val_logged = False

    def forward(self, x) -> Any:
        return self._generator(x)

    def configure_optimizers(self) -> Any:
        if self._generator_scheduler and self._discriminator_scheduler:
            return [
                self._generator_optimizer, self._discriminator_optimizer
            ], [
                self._generator_scheduler, self._discriminator_scheduler
            ]
        return [self._generator_optimizer, self._discriminator_optimizer]

    def training_step(self, batch, batch_idx, optimizer_idx) -> STEP_OUTPUT:
        target, mask = batch

        # train gen
        if optimizer_idx == 0:
            output = self._generator(target*mask)
            ones = torch.ones(target.shape[0], 1, 1, 1, dtype=target.dtype, device=target.device)
            
            mse_loss = self._mse_loss(output*mask, target*mask)
            generator_loss = self._bce_loss(self._discriminator(output), ones)
            self.log("train/mse_loss", mse_loss)
            self.log("train/generator_loss", generator_loss)
            return {
                "loss": self._alpha*generator_loss + mse_loss
            }
            
        # train discriminator
        if optimizer_idx == 1:
            # real samples
            ones = torch.ones([target.shape[0], 1, 1, 1], dtype=target.dtype, device=target.device)
            real_bce_loss = self._bce_loss(
                self._discriminator(target), ones
            )  

            # fake samples
            zeros = torch.zeros([target.shape[0], 1, 1, 1], dtype=target.dtype, device=target.device)
            fake_bce_loss = self._bce_loss(
                self._discriminator(self._generator(target*mask).detach()), zeros
            )
            discriminator_loss = (real_bce_loss + fake_bce_loss) / 2
            self.log("train/discriminator_loss", discriminator_loss)
            return {
                "loss": discriminator_loss
            }

        raise RuntimeError(f"Received unhandled optimizer index {optimizer_idx}.")

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        if not self._val_logged:
            target, mask = batch
            output = self._generator(target*mask)

            self.logger.experiment.add_images('val/target', target, self.global_step)
            self.logger.experiment.add_images('val/masked_target', target*mask, self.global_step)
            self.logger.experiment.add_images('val/output', output, self.global_step)
            self._val_logged = True

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self._val_logged = False
        return super().validation_epoch_end(outputs)
