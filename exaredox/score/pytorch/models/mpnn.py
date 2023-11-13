# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import torch
from torch import nn
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import MessagePassing

from ..data import Molecule
from .base import EncoderNoCoords

__all__ = ["MPNN"]


class MPNNConv(MessagePassing):
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.edge_mlp = nn.Linear(input_dim, hidden_dim * input_dim, bias=bias)
        self.node_mlp = nn.Linear(hidden_dim, input_dim, bias=bias)
        self.input_dim = input_dim

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        edge_feats = self.edge_mlp(edge_attr)
        edge_feats = edge_feats.view(-1, self.input_dim, self.input_dim)
        return torch.matmul(x_j.unsqueeze(1), edge_feats).squeeze(1)

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        new_feats = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        outputs = new_feats + self.node_mlp(x)
        return outputs


class MPNNBlock(nn.Module):
    def __init__(self, norm: str | bool | None = None, **conv_kwargs):
        super().__init__()
        # confusingly named, but this refers to the output of the convolution
        # which currently matches the input dim for residual connection
        output_dim = conv_kwargs.get("input_dim")
        if norm is None:
            norm = nn.Identity()
        if isinstance(norm, bool):
            if norm:
                norm = nn.BatchNorm1d(output_dim)
            else:
                norm = nn.Identity()
        if isinstance(norm, str):
            norm_class = getattr(nn, norm, None)
            if not norm_class:
                raise NameError(
                    f"{norm} requested as normalization module"
                    "but not found in torch.nn",
                )
            norm = norm_class(output_dim)
        self.norm = norm
        self.conv = MPNNConv(**conv_kwargs)

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        node_feats = self.conv(x, edge_index, edge_attr)
        node_feats = self.norm(node_feats)
        return node_feats


class MPNN(EncoderNoCoords):
    def __init__(
            self,
            hidden_dim: int,
            output_dim: int,
            num_conv: int = 3,
            num_atom_types: int = 100,
            num_edge_types: int = 30,
            norm: bool = False,
    ) -> None:
        super().__init__()
        self.atom_embeddings = nn.Embedding(num_atom_types, hidden_dim, padding_idx=0)
        self.edge_embeddings = nn.Embedding(num_edge_types, hidden_dim, padding_idx=0)
        self.output = nn.Linear(hidden_dim, output_dim)
        conv_kwargs = {
            "input_dim": hidden_dim,
            "hidden_dim": hidden_dim,
        }
        self.conv_layers = nn.ModuleList(
            [MPNNBlock(norm, **conv_kwargs) for _ in range(num_conv)],
        )

    def _forward(
            self,
            batch: Molecule,
            **kwargs,
    ) -> torch.Tensor:
        super()._forward(batch)
        atoms = batch.atoms
        bonds = batch.bonds
        edge_index = batch.edge_index
        node_batch_idx = getattr(batch, "batch", None)
        # embedding table for atomic numbers
        node_feats = self.atom_embeddings(atoms)
        # embed bonds based on bond order?
        edge_feats = self.edge_embeddings(bonds)
        for layer in self.conv_layers:
            node_feats = layer(node_feats, edge_index, edge_feats)
        # size extensive property prediction
        readout = global_add_pool(node_feats, node_batch_idx)
        model_outputs = self.output(readout)
        return model_outputs
