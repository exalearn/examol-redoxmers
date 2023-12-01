# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import torch
from einops import reduce
from torch import nn
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import MessagePassing

from torch_geometric.typing import Size

from .base import CoordinateNormalization, EncoderWithCoords
from .utils import make_mlp

"""
Some inspiration from https://github.com/lucidrains/egnn-pytorch but
implementation was otherwise from scratch by Kelvin Lee, following
Satorras, Hoogeboom, Welling (2022).
"""


class EGNNConv(MessagePassing):
    """
    Implements a single E(n)-GNN convolution layer, or in Satorras _et al._
    referred to as "Equivariant Graph Convolutional Layer" (EGCL).

    One modification to the architecture is the addition of ``LayerNorm``
    in the messages.
    """

    def __init__(
            self,
            node_dim: int,
            hidden_dim: int = 64,
            out_dim: int = 1,
            coord_dim: int | None = None,
            edge_dim: int | None = None,
            activation: str = "SiLU",
            num_layers: int = 2,
            norm_coords: bool = True,
            norm_edge_feats: bool = True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        # default use same dimensionality as the node ones to make things simpler
        if not edge_dim:
            edge_dim = node_dim
        if not coord_dim:
            coord_dim = node_dim
        # two sets of node features for ij, relative distance, and edge features
        self.edge_mlp = make_mlp(
            node_dim * 2 + 1 + edge_dim,
            hidden_dim,
            out_dim,
            activation,
            num_layers,
        )
        # include layer norm to the messages
        if norm_edge_feats:
            self.edge_norm = LayerNorm(hidden_dim)
        else:
            self.edge_norm = nn.Identity()
        # this transforms embeds coordinates
        self.coord_mlp = make_mlp(
            coord_dim,
            hidden_dim,
            coord_dim,
            activation="SiLU",
            bias=False,
        )
        self.edge_projection = nn.Linear(out_dim, 1, bias=False)
        if norm_coords:
            self.coord_norm = CoordinateNormalization()
        else:
            self.coord_norm = nn.Identity()
        self.node_mlp = make_mlp(
            out_dim + node_dim,
            hidden_dim,
            out_dim,
            activation="SiLU",
        )

    def message(self, atom_feats_i, atom_feats_j, edge_attr) -> torch.Tensor:
        # coordinate distances already included as edge_attr
        joint = torch.cat([atom_feats_i, atom_feats_j, edge_attr], dim=-1)
        edge_feats = self.edge_mlp(joint)
        return self.edge_norm(edge_feats)

    def propagate(
            self,
            edge_index: torch.Tensor,
            size: Size | None = None,
            **kwargs,
    ):
        size = self._check_input(edge_index, size)
        kwarg_dict = self._collect(self._user_args, edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute("message", kwarg_dict)
        agg_kwargs = self.inspector.distribute("aggregate", kwarg_dict)
        update_kwargs = self.inspector.distribute("update", kwarg_dict)

        # pull out some of the expected arguments
        coords = kwargs.get("coords")  # shape [N, 3]
        atom_feats = kwargs.get("atom_feats")  # shape [N, node_dim]
        rel_coords = kwargs.get("rel_coords")  # shape [E, 3]

        # eq 3, calculate messages along edges
        msg_ij = self.message(**msg_kwargs)
        edge_weights = self.edge_projection(msg_ij)

        # eq 5, aggregated messages
        hidden_nodes = self.aggregate(msg_ij, **agg_kwargs)

        # eq 4, add wewighted sum to coordinates
        num_edges = edge_index.size(1)
        edge_norm_factor = 1 / (num_edges - 1)
        weighted_distances = edge_norm_factor * self.aggregate(
            rel_coords * edge_weights,
            **agg_kwargs,
        )
        # now update the coordinates
        new_coords = self.coord_norm(coords + weighted_distances)

        # eq 6, transform node features
        new_node_feats = self.node_mlp(torch.cat([hidden_nodes, atom_feats], dim=-1))
        return self.update((new_node_feats, new_coords), **update_kwargs)

    def forward(
            self,
            atom_feats: torch.Tensor,
            coords: torch.Tensor,
            edge_feats: torch.Tensor,
            edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # becomes shape [num_edges, 3]
        rel_coords = coords[edge_index[0]] - coords[edge_index[1]]
        # basically sum of squares
        rel_dist = reduce(rel_coords.square(), "edges xyz -> edges ()", "sum")
        combined_edge_feats = torch.cat([edge_feats, rel_dist], dim=-1)
        new_nodes, new_coords = self.propagate(
            edge_index,
            atom_feats=atom_feats,
            coords=coords,
            rel_coords=rel_coords,
            edge_attr=combined_edge_feats,
        )
        return new_nodes, new_coords


class EGNN(EncoderWithCoords):
    """EGNN

    Args:
        hidden_dim: Number of features to describe each node and edge
        output_dim: Number of outputs to produce
        num_conv: Number of message passing layers
        num_atom_types: Maximum number of types of nodes to expect (0 is a mask)
        num_edge_types: Maximum number of types of edges expect (0 is a mask)
        activation: Activation function to use in message passing teps
        pool_operation: Name of the pool operation to use (ex: "add" for "global_add_pool")
        num_output_layers: Number of output layers
        pool_before_output: Whether to pool before or after applying the output layers
        mlp_per_output: Train a single MLP which maps network or node features for each output
            instead of a single, multi-output MLP
    """

    def __init__(
            self,
            hidden_dim: int = 64,
            num_conv: int = 3,
            num_atom_types: int = 100,
            num_edge_types: int = 30,
            activation: str = "SiLU",
            **kwargs,
    ) -> None:
        super().__init__(hidden_dim=hidden_dim, **kwargs)
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim, padding_idx=0)
        self.edge_embedding = nn.Embedding(num_edge_types, hidden_dim, padding_idx=0)
        # embeds coordinates as part of EGNN
        self.coord_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim, bias=False),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        # generate sequence of EGNN convolution layers
        self.conv_layers = nn.ModuleList(
            [
                EGNNConv(
                    hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    activation=activation,
                    **kwargs,
                )
                for _ in range(num_conv)
            ],
        )
