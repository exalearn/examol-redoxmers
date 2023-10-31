"""Interfaces to models produced by @laserkelvin"""
from io import BytesIO
from tempfile import TemporaryDirectory
from typing import Sequence

import torch
import numpy as np
from future.moves.itertools import zip_longest
from lightning import pytorch as pl
from torch_geometric.loader import DataLoader as PyGLoader

from redox_models import RedoxDataModule, RedoxTask
from sklearn.model_selection import train_test_split

from examol.score.base import Scorer, collect_outputs
from examol.simulate.initialize import add_initial_conformer
from examol.store.models import MoleculeRecord
from examol.store.recipes import PropertyRecipe
from redox_models.data import RedoxData

ModelParamType = tuple[str, dict[str, object]]
ModelObjectType = tuple[ModelParamType, bytes | None]


class RedoxModelsScorer(Scorer):
    """Wrapper for Kelvin's RedoxModels.

    Users provide a list of models as model architecture name (e.g., MPNN) a set of hyperparameters paired with, optionally, an initial set of weights.

    .. code-block: python

        models = [
            (('MPNN', dict(hidden_dim=32, output_dim=1), None)
        ]

    Passes molecules to the remote system as tuple of an XYZ format string of the molecule geometry computed at specified level,
    and either a list of known values of the target property if using multi-fidelity or ``None`` otherwise.

    Transmits the model as a byte string that was serialized using ``torch.save``.

    Args:
        conformer: Name of the source and charge for the conformer.
            Default is to use MMFF energies for neutral geometries.
    """

    _supports_multi_fidelity = True

    def __init__(self, conformer: tuple[str, int] = ('mmff', 0)):
        self.conformer = conformer

    def transform_inputs(self, record_batch: list[MoleculeRecord], recipes: Sequence[PropertyRecipe] | None = None) -> list[tuple[str, None | list[float]]]:
        output = []
        config_name, charge = self.conformer
        for record in record_batch:
            # Get the molecule
            add_initial_conformer(record)  # In case it has not been done yet
            conf, _ = record.find_lowest_conformer(config_name, charge, solvent=None, optimized_only=True)

            # Get the target values
            known = None if recipes is None else collect_outputs([record], recipes).tolist()
            output.append((conf.xyz, known))
        return output

    def prepare_message(self, model: ModelObjectType, training: bool = True) -> (tuple[str, dict[str, object]] | bytes):
        if training:
            return model[0]
        else:
            return model[1]

    def retrain(self, model_msg: tuple[str, dict[str, object]], inputs: list, outputs: list, **kwargs) -> bytes:
        # Make the data loader
        train_data, val_data = train_test_split(list(zip(inputs, outputs)), test_size=0.1)
        data_module = RedoxDataModule(
            train_path=train_data,
            val_path=val_data,
        )

        # Run the training in a temporary directory
        with TemporaryDirectory() as tmpdir:
            model, kwargs = model_msg
            task = RedoxTask(
                model,
                kwargs,
                lr=1e-3,
                weight_decay=0.0,
            )
            trainer = pl.Trainer(max_epochs=2,
                                 default_root_dir=tmpdir,
                                 enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)  # No print to screen
            trainer.fit(task, datamodule=data_module)

            # Return the model as a serialized object
            fp = BytesIO()
            torch.save(task.encoder, fp)
            return fp.getvalue()

    def update(self, model: ModelObjectType, update_msg: bytes) -> ModelObjectType:
        return model[0], update_msg

    def score(self, model_msg: bytes, inputs: list, **kwargs) -> np.ndarray:
        # Unpack the model
        model = torch.load(BytesIO(model_msg), map_location='cpu')

        # Make the data loader
        with_targets = zip_longest(inputs, '', fillvalue=None)
        data_module = RedoxData(with_targets)
        loader = PyGLoader(data_module)

        # Run over all data
        outputs = []
        for batch in loader:
            outputs.append(model(batch).detach().cpu().numpy())
        return np.concatenate(outputs, axis=0)

