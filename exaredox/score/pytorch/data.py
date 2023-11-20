# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

"""Data structures used by PyTorch models"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

import torch
import numpy as np
from lightning import pytorch as pl
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdchem import BondType
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from torch_geometric.loader import DataLoader as PyGLoader

from .utils import MeanScaler

__all__ = ["Molecule", "RedoxData", "RedoxDataModule", "get_graph_information"]

RecordType = dict[str, int | str | np.ndarray]  # Graph data mapped to one or more floats

pt = Chem.GetPeriodicTable()

bond_names = sorted(filter(lambda x: x.isupper(), dir(BondType)))
# The bond mapping offsets by 1 to facilitate embedding padding
bond_mapping = {
    getattr(BondType, name): index + 1 for index, name in enumerate(bond_names)
}


def get_graph_information(xyz) -> dict[str, int | str | np.ndarray]:
    """Convert an XYZ file to a PyG-compatible dictionary

    Args:
        xyz: XYZ to process
    Returns:
        Dictionary containing the graph information
    """

    mol = Chem.MolFromXYZBlock(xyz)
    rdDetermineBonds.DetermineConnectivity(mol)
    rdDetermineBonds.DetermineBonds(mol)
    bonds = []
    edge_index = []
    for bond in mol.GetBonds():
        # encode the bond order
        bond_type = bond.GetBondType()
        bond_encoding = bond_mapping[bond_type]
        bonds.append(bond_encoding)
        # now get src and dest
        src = bond.GetBeginAtomIdx()
        dst = bond.GetEndAtomIdx()
        edge_index.append([src, dst])

    # get atom types
    atom_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    coords = mol.GetConformer(0).GetPositions().astype(np.float32)
    return {
        "atoms": np.array(atom_numbers, dtype=np.int64),
        "edge_index": np.array(edge_index, dtype=np.int64).T,
        "bonds": np.array(bonds, dtype=np.int64),
        "coords": coords,
        "num_nodes": len(atom_numbers),
        "smi": Chem.MolToSmiles(mol, canonical=True),
    }


def parse_xyz(coordinates: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Get atom types and coordinates from XYZ file

    Args:
        coordinates: XYZ-format structure
    Returns:
        - Atom types
        - Coordinates
    """
    # skip the first two lines, and the last line
    lines = coordinates.split("\n")[2:-1]
    num_atoms = len(lines)
    atoms = torch.zeros(num_atoms, dtype=torch.long)
    coords = torch.zeros(num_atoms, 3, dtype=torch.float64)
    for index, line in enumerate(lines):
        split_line = line.split()
        symbol = split_line.pop(0)
        atoms[index] = pt.GetAtomicNumber(symbol)
        for coord_index, value in enumerate(split_line):
            coords[index, coord_index] = float(value)
    return atoms, coords


def compute_edges(coords: torch.Tensor, cutoff: float = 10.0) -> torch.Tensor:
    """Assign edges between all atoms within a distance from each other

    Args:
        coords: Coordinates of atoms
        cutoff: Cutoff distance (units: Ang)
    Returns:
        Indices of edges
    """
    # expect coords to be shape N, 3
    dist_mat = torch.cdist(coords, coords)
    lower_triangle = torch.tril(dist_mat)
    mask = (0 < lower_triangle) * (lower_triangle <= cutoff)
    edge_index = torch.argwhere(mask)
    if edge_index.size(0) != 2:
        edge_index = edge_index.T
    # now try and get common indices grouped together
    edge_index = edge_index[torch.argsort(edge_index[:, 0])].contiguous()
    return edge_index


class Molecule(Data):
    def __init__(
            self,
            atoms: torch.LongTensor,
            edge_index: torch.LongTensor,
            bonds: torch.LongTensor,
            target: float | Iterable[float],
            num_nodes: int,
            coords: torch.FloatTensor | None,
            smi: str | None,
    ) -> None:
        super().__init__(
            atoms=atoms,
            bonds=bonds,
            edge_index=edge_index,
            coords=coords,
            target=target,
            num_nodes=num_nodes,
            smi=smi,
        )

    @classmethod
    def from_xyz_and_target(cls, xyz: str, target: float | Iterable[float]):
        """Create the Molecule record from XYZ and property

        Args:
            xyz: XYZ structure to process
            target: Target value to predict
        Returns:
            Initialized record, ready to train
        """
        graph_data = get_graph_information(xyz)
        return cls.from_graph_data(graph_data, target)

    @classmethod
    def from_graph_data(cls, graph_data: dict[str, float | int | np.ndarray], target: float | Iterable[float]):
        """Create the Molecule record from pre-parsed graph data

        Args:
            graph_data: Data about the molecule graph
            target: Target value to predict
        Returns:
            Initialized record as a PyTorch object
        """
        as_torch = {}
        for t in ['atoms', 'bonds', 'edge_index', 'coords']:  # Things to convert to Torch now
            as_torch[t] = torch.from_numpy(graph_data[t])
        for t in ['num_nodes', 'smi']:
            as_torch[t] = graph_data[t]

        return cls(target=target, **as_torch)


class RedoxData(Dataset):
    def __init__(
            self,
            data: list[RecordType],
            precomputed: bool = False,
            transforms: list[Callable] | None = None,
    ) -> None:
        super().__init__()
        # Store a copy of the data
        self._data = [Molecule.from_graph_data(info, target) for info, target in data]

        # Store any transformations
        self.precomputed = precomputed
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Molecule:
        sample = self._data[index]

        # apply sequence of transforms if specified
        if self.transforms:
            sample = sample.clone()  # Do not edit the original
            for transform in self.transforms:
                sample = transform(sample)
        return sample

    @property
    def scaler(self) -> MeanScaler | None:
        if self.transforms:
            for transform in self.transforms:
                if isinstance(transform, MeanScaler):
                    return transform
        return None

    @staticmethod
    def collate_func(batch: list[Molecule]) -> Molecule:
        out, _, _ = collate(Molecule, batch)
        return out


class RedoxDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_data: list[RecordType] = None,
            valid_data: list[RecordType] = None,
            test_data: list[RecordType] = None,
            predict_data: list[RecordType] = None,
            transforms: list[Callable] | None = None,
            **loader_kwargs,
    ) -> None:
        super().__init__()
        loader_kwargs.setdefault("batch_size", 16)
        loader_kwargs.setdefault("num_workers", 4)

        # Set the hyperparameters
        self.loader_kwargs = loader_kwargs
        self.transforms = transforms

        # Load the data
        self.datasets: dict[str, RedoxData] = dict()
        for name, data in zip(['train', 'val', 'test', 'predict'], [train_data, valid_data, test_data, predict_data]):
            if data is not None:
                self.datasets[name] = RedoxData(data, transforms=self.transforms)

        self.save_hyperparameters(ignore=['datasets', 'train_data', 'valid_data', 'test_data', 'predict_data'])

    @property
    def persistent_workers(self) -> bool:
        return True if self.loader_kwargs["num_workers"] > 0 else False

    def _make_dataloader(self, target: str) -> PyGLoader:
        shuffle = True if target == "train" else False
        dset = self.datasets[target]
        kwargs = {
            "batch_size": self.loader_kwargs.get("batch_size"),
            "num_workers": self.loader_kwargs.get("num_workers"),
            "persistent_workers": self.persistent_workers,
            "shuffle": shuffle,
            "collate_fn": dset.collate_func,
        }
        return PyGLoader(dset, **kwargs)

    @property
    def scaler(self) -> MeanScaler | None:
        if self.transforms:
            for transform in self.transforms:
                if isinstance(transform, MeanScaler):
                    return transform
        return None

    def train_dataloader(self):
        loader = self._make_dataloader("train")
        return loader

    def val_dataloader(self):
        loader = self._make_dataloader("val")
        return loader

    def test_dataloader(self):
        loader = self._make_dataloader("test")
        return loader

    def predict_dataloader(self):
        loader = self._make_dataloader("predict")
        return loader
