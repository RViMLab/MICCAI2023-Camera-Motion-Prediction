from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl


class WorstSamplingCallback(pl.Callback):
    def __init__(self, worst: float=0.1, random: float=0.4) -> None:
        if worst + random > 1.:
            raise ValueError(f"Invalid values, worst + random must be less or equal 1, got {worst} + {random} = {worst + random}.")
        self._per_sequence_loss_buffer = pd.DataFrame(columns=["loss"])
        self._worst = worst
        self._random = random
        super().__init__()

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        batch_size = trainer.train_dataloader.loaders.batch_size
        batch_idcs = (np.arange(batch_size) + batch_idx*batch_size).tolist()
        sample_idcs = trainer.train_dataloader.dataset.datasets.sample_idcs[batch_idcs]

        # build currently best losses
        if len(self._per_sequence_loss_buffer) != len(trainer.train_dataloader.dataset.datasets.valid_idcs):
            per_sequence_loss_buffer = pd.DataFrame(
                outputs["per_sequence_loss"].tolist(),
                index=pd.Index(sample_idcs, "int64"),
                columns=["loss"]
            )
            self._per_sequence_loss_buffer = pd.concat([
                per_sequence_loss_buffer, self._per_sequence_loss_buffer
            ], axis=0)
        else:
            self._per_sequence_loss_buffer.loc[sample_idcs, "loss"] = outputs["per_sequence_loss"].tolist()

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # get worst and random indices
        worst = int(len(self._per_sequence_loss_buffer)*self._worst)
        random = int(len(self._per_sequence_loss_buffer)*self._random)
        sorted_buffer = self._per_sequence_loss_buffer.sort_values("loss", ascending=False)
        sample_idcs = sorted_buffer.iloc[:worst].index
        sample_idcs = sample_idcs.append(sorted_buffer.iloc[worst:].sample(random).index)
        
        # update dataset
        trainer.train_dataloader.dataset.datasets.sample_idcs = sample_idcs
        return super().on_train_epoch_end(trainer, pl_module)
