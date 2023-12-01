# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License
from abc import ABC
from abc import abstractmethod

import torch
from torch import nn
from torch_geometric.nn import pool

from ..data import Molecule
from .utils import make_mlp


class AbstractEncoder(ABC, nn.Module):
    conv_layers: nn.ModuleList
    """Convolution layers"""
    output_mlp: nn.ModuleList | nn.Module
    """Output layers which convert atom features to outputs"""

    def __init__(self,
                 output_dim: int = 1,
                 hidden_dim: int = 64,
                 pool_operation: str = 'add',
                 num_output_layers: int = 0,
                 mlp_per_output: bool = True,
                 pool_before_output: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.pool_operation = getattr(pool, f'global_{pool_operation}_pool')
        self.pool_before_output = pool_before_output
        self.mlp_per_output = mlp_per_output and output_dim > 1
        if self.mlp_per_output:
            self.output_mlp = nn.ModuleList([
                make_mlp(hidden_dim, hidden_dim, 1, activation="ReLU", num_layers=num_output_layers)
                for _ in range(output_dim)
            ])
        else:
            self.output_mlp = make_mlp(hidden_dim, hidden_dim, output_dim, activation="ReLU", num_layers=num_output_layers)

    def forward(self, batch: Molecule):
        # Run the message passing steps
        atom_feats = self._forward(batch)

        # Condense from atom features to one output per graph
        node_batch_idx = getattr(batch, "batch", None)

        def _run_output(data):
            if self.mlp_per_output:
                outputs = [f(data) for f in self.output_mlp]
                return torch.concat(outputs, dim=-1)
            else:
                return self.output_mlp(data)

        if self.pool_before_output:
            pooled_data = self.pool_operation(atom_feats, node_batch_idx)
            return _run_output(pooled_data)
        else:
            output_per_node = _run_output(atom_feats)
            return self.pool_operation(output_per_node, node_batch_idx)

    @abstractmethod
    def _forward(self, batch, *args, **kwargs) -> torch.Tensor:
        """Produce features for each atom in the network"""
        raise NotImplementedError()


class EncoderNoCoords(AbstractEncoder):
    def _forward(self, batch, *args, **kwargs) -> torch.Tensor:
        for key in ["atoms", "bonds", "edge_index"]:
            assert key in batch, f"Expected {key} in batch, but was not found: {batch}"
        atoms = batch.atoms
        bonds = batch.bonds
        edge_index = batch.edge_index

        # embed coordinates, then lookup embeddings for atoms and bonds
        atom_feats = self.atom_embedding(atoms)
        edge_feats = self.edge_embedding(bonds)

        # loop over each graph layer
        for layer in self.conv_layers:
            atom_feats = layer(atom_feats, edge_index, edge_feats)
        return atom_feats


class EncoderWithCoords(AbstractEncoder):
    def _forward(self, batch, *args, **kwargs) -> torch.Tensor:
        for key in ["atoms", "bonds", "edge_index", "coords"]:
            assert key in batch, f"Expected {key} in batch, but was not found: {batch}"

        atoms = batch.atoms
        bonds = batch.bonds
        edge_index = batch.edge_index
        coords = batch.coords

        # embed coordinates, then lookup embeddings for atoms and bonds
        coords = self.coord_embedding(coords)
        atom_feats = self.atom_embedding(atoms)
        edge_feats = self.edge_embedding(bonds)

        # loop over each graph layer
        for layer in self.conv_layers:
            atom_feats, coords = layer(atom_feats, coords, edge_feats, edge_index)
        return atom_feats


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
