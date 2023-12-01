# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from __future__ import annotations

import torch
from torch import nn


class ResidualConnection(nn.Module):
    def __init__(self, is_active: bool = True) -> None:
        super().__init__()
        self.is_active = is_active

    def forward(
            self,
            prior_feats: torch.Tensor,
            new_feats: torch.Tensor,
    ) -> torch.Tensor:
        if self.is_active:
            new_feats.add_(prior_feats)
        return new_feats


def make_mlp(
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        activation: str | None,
        num_layers: int = 2,
        **kwargs,
) -> nn.Module:
    if not activation:
        activation = nn.Identity
    else:
        activation = getattr(nn, activation, None)
        if not activation:
            raise NameError(
                f"Requested activation {activation}, but not found in torch.nn",
            )
    if num_layers > 0:
        layers = [nn.Linear(in_dim, hidden_dim, **kwargs), activation()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim, **kwargs), activation()])
        layers.append(nn.Linear(hidden_dim, out_dim, **kwargs))
        return nn.Sequential(*layers)
    else:
        return nn.Linear(in_dim, out_dim, **kwargs)
