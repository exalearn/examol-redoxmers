# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

"""Data structures used by PyTorch models"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any
from typing import Callable

import torch
from lightning import pytorch as pl
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdchem import BondType
from torch.utils.data import DataLoader as NativeLoader
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from torch_geometric.loader import DataLoader as PyGLoader

from .transforms import MeanScaler

__all__ = ["Molecule", "RedoxData", "RedoxDataModule", "compute_edges"]

RecordType = tuple[str, float | list[float] | None]  # XYZ mapped to one or more flats

pt = Chem.GetPeriodicTable()

bond_names = sorted(filter(lambda x: x.isupper(), dir(BondType)))
# The bond mapping offets by 1 to facilitate embedding padding
bond_mapping = {
    getattr(BondType, name): index + 1 for index, name in enumerate(bond_names)
}


def get_graph_information(xyz) -> dict[str, Any]:
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
    coords = torch.from_numpy(mol.GetConformer(0).GetPositions()).float()
    return_dict = {
        "atoms": torch.LongTensor(atom_numbers),
        "edge_index": torch.LongTensor(edge_index).T.contiguous(),
        "bonds": torch.LongTensor(bonds),
        "coords": coords,
        "num_nodes": len(atom_numbers),
        "smi": Chem.MolToSmiles(mol, canonical=True),
    }
    return return_dict


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
            coords=coords,
            target=target,
            edge_index=edge_index,
            bonds=bonds,
            num_nodes=num_nodes,
            smi=smi,
        )

    @classmethod
    def from_xyz_and_target(cls, xyz: str, target_property: float | Iterable[float]):
        """Create the Molecule record from XYZ and property

        Args:
            xyz: XYZ structure to process
            target_property: Target value to predict
        Returns:
            Initialized record, ready to train
        """
        graph_data = get_graph_information(xyz)
        return cls(target=target_property, **graph_data)


class RedoxData(Dataset):
    def __init__(
            self,
            data: list[RecordType],
            precomputed: bool = False,
            transforms: list[Callable] | None = None,
    ) -> None:
        super().__init__()
        # Store a copy of the data
        self._data = [Molecule.from_xyz_and_target(xyz, target) for (xyz, _), target in data]

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


class RedoxPointCloud(RedoxData):
    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        graph = super().__getitem__(index)
        # unpack into a point cloud now
        data = {
            "coords": graph.coords,
            "atoms": graph.atoms,
            "smiles": graph.smi,
            "target": graph.target,
        }
        return data

    @staticmethod
    def collate_func(
            batch: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor | list[str]]:
        batch_size = len(batch)
        max_atoms = 0
        for entry in batch:
            atom_count = len(entry["atoms"])
            if atom_count > max_atoms:
                max_atoms = atom_count
        padded_atoms = torch.zeros((batch_size, max_atoms), dtype=torch.long)
        padded_coords = torch.zeros((batch_size, max_atoms, 3))
        mask = torch.zeros_like(padded_atoms).bool()
        target = torch.zeros((batch_size,))
        smiles = []
        for index, entry in enumerate(batch):
            atoms = entry["atoms"]
            atom_count = len(atoms)
            padded_atoms[index, :atom_count] = atoms
            padded_coords[index, :atom_count, :] = entry["coords"]
            mask[index, :atom_count] = True
            smiles.append(entry["smiles"])
            target[index] = entry["target"]
        packed_data = {
            "atoms": padded_atoms,
            "coords": padded_coords,
            "mask": mask,
            "smiles": smiles,
            "target": target,
        }
        return packed_data


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
