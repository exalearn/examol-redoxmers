# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License
from abc import ABC
from abc import abstractmethod

import torch
from torch import nn

from ..data import Molecule


class AbstractEncoder(ABC, nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, batch: Molecule):
        # unpack data into model
        return self._forward(batch)

    @abstractmethod
    def _forward(self, batch, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()


class EncoderNoCoords(AbstractEncoder):
    def _forward(self, batch, *args, **kwargs) -> torch.Tensor:
        for key in ["atoms", "bonds", "edge_index"]:
            assert key in batch, f"Expected {key} in batch, but was not found: {batch}"
        ...


class EncoderWithCoords(AbstractEncoder):
    def _forward(self, batch, *args, **kwargs) -> torch.Tensor:
        for key in ["atoms", "bonds", "edge_index", "coords"]:
            assert key in batch, f"Expected {key} in batch, but was not found: {batch}"
        ...


class PointCloudEncoder(AbstractEncoder):
    def _forward(self, batch, *args, **kwargs) -> torch.Tensor:
        for key in ["atoms", "coords"]:
            assert key in batch, f"Expected {key} in batch, but was not found: {batch}"
        ...


class CoordinateNormalization(nn.Module):
    def __init__(self, epsilon: float = 1e-7) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coord_norm = coords.norm(dim=-1, keepdim=True)
        # rescale coordinates by the norm, then rescale by learnable scale
        new_coords = (coords / (coord_norm + self.epsilon)) * self.scale
        return new_coords
