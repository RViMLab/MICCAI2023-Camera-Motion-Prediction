import pytorch_lightning as pl
from typing import List
import warnings
import bisect


class RhoCallback(pl.Callback):
    r"""Increases the random edge homography edge deviation over epochs.

    Args:
        rhos (List[int]): Edge deviation interval in pixels
        epochs (List[int]): Epochs with count starting from 0
    """
    def __init__(self, rhos: List[int], epochs: List[int]):
        if len(rhos) is not len(epochs):
            raise ValueError('Length of rhos {} must equal length of epochs {}'.format(len(rhos), len(epochs)))
        self._rhos = rhos
        self._epochs = epochs

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.max_epochs > max(self._epochs):
            warnings.warn("Epoch change list tries to callback at epoch {} greater than trainer's max epoch  {}".format(max(self._epochs), trainer.max_epochs), UserWarning)

        idx = bisect.bisect_right(self._epochs, trainer.current_epoch) - 1  # lightning indexing [0, ..., N-1] epochs
        trainer.datamodule.rho = self._rhos[idx]
