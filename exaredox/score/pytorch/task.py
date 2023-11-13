# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

"""Definition of a training task"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from lightning import pytorch as pl
from torch import nn
from torch.optim import AdamW

from . import models
from .data import Molecule


class RedoxTask(pl.LightningModule):
    """Module used for a Redox module"""

    def __init__(
        self,
        encoder_class: str | nn.Module,
        encoder_kwargs: Mapping[str, Any] | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        loss_func: str | nn.Module = "MSELoss",
    ) -> None:
        super().__init__()
        if isinstance(encoder_class, str):
            encoder_class = getattr(models, encoder_class, None)
            if not encoder_class:
                raise NameError(
                    f"Encoder class expected {encoder_class} but not found in models.",
                )
        self.encoder = encoder_class(**encoder_kwargs)
        if isinstance(loss_func, str):
            loss_func = getattr(nn, loss_func, None)
            if not loss_func:
                raise NameError(
                    f"{loss_func} was requested as a loss function"
                    "but not found in torch.nn",
                )
            loss_func = loss_func()
        self.loss_func = loss_func
        self.save_hyperparameters(ignore='loss_func')

    def configure_optimizers(self) -> AdamW:
        opt = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return opt

    def forward(self, batch: Molecule) -> torch.Tensor:
        return self.encoder(batch)

    def _abstract_step(self, batch: Molecule | dict[str, Any], stage: str) -> float:
        """
        Implements an abstract step that can be reused at other stages.

        This is relatively simple and allows reuse between training, validation,
        and testing loops as all we are doing is doing property regression for
        a single target.

        Parameters
        ----------
        batch : Molecule | dict[str, Any]
            Batch of molecules, either as a batched ``Molecule`` object
            or a dictionary comprising concatenated tensors for point clouds.
        stage : str
            Name of the stage calling this step. Used for logging annotation.

        Returns
        -------
        float
            Value of the loss
        """
        pred_y = self(batch)
        # if it's just a scalar target, just flatten the data
        if pred_y.size(-1) == 1:
            pred_y.squeeze_()
        if isinstance(batch, dict):
            targets = batch.get("target")
        else:
            targets = batch.target
        loss = self.loss_func(pred_y, targets)
        batch_size = getattr(batch, "num_graphs", 1)
        self.log(
            f"{stage}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def training_step(self, batch: Molecule, *args, **kwargs) -> float:
        loss = self._abstract_step(batch, "train")
        return loss

    def validation_step(self, batch: Molecule, *args, **kwargs) -> float:
        loss = self._abstract_step(batch, "val")
        return loss

    def test_step(self, batch: Molecule, *args, **kwargs) -> float:
        loss = self._abstract_step(batch, "test")
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self(batch)
