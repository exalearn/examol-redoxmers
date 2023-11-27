"""Interfaces to models produced by @laserkelvin"""
from io import BytesIO
from tempfile import TemporaryDirectory

import torch
import numpy as np
from future.moves.itertools import zip_longest
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from torch.nn import MSELoss
from torch_geometric.loader import DataLoader as PyGLoader
from sklearn.model_selection import train_test_split

from examol.score.base import MultiFidelityScorer
from examol.score.utils.multifi import compute_deltas
from examol.simulate.initialize import add_initial_conformer
from examol.store.models import MoleculeRecord

from .data import RedoxData, RedoxDataModule, get_graph_information
from .task import RedoxTask
from .utils import MaskedMSELoss, MeanScaler

ModelParamType = tuple[str, dict[str, object]]
ModelObjectType = tuple[ModelParamType, bytes | None]
ModelRecordType = dict[str, int | float | np.ndarray]


class RedoxModelsScorer(MultiFidelityScorer):
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

    def __init__(self, conformer: tuple[str, int] = ('mmff', 0)):
        self.conformer = conformer

    def transform_inputs(self, record_batch: list[MoleculeRecord]) -> list[ModelRecordType]:
        output = []
        config_name, charge = self.conformer
        for record in record_batch:
            # Get the molecule
            add_initial_conformer(record)  # In case it has not been done yet
            conf, _ = record.find_lowest_conformer(config_name, charge, solvent=None, optimized_only=False)

            # Get the target values
            output.append(get_graph_information(conf.xyz))
        return output

    def prepare_message(self, model: ModelObjectType, training: bool = True) -> (ModelParamType | bytes):
        if training:
            return model[0]
        else:
            return model[1]

    def retrain(self,
                model_msg: ModelParamType,
                inputs: list[ModelRecordType],
                outputs: list[float] | np.ndarray,
                lower_fidelities: np.ndarray | None = None,
                max_epochs: int = 2,
                patience: int | None = None,
                batch_size: int = 32,
                learning_rate: float = 1e-3,
                validation_size: float = 0.1,
                num_workers: int = 4,
                verbose: bool = False) -> bytes:

        # Compute deltas if lower_fidelities are provided
        if lower_fidelities is not None:
            all_fidelities = np.concatenate([lower_fidelities, np.array(outputs)[:, None]], axis=1)
            outputs = compute_deltas(all_fidelities)
            loss_func = MaskedMSELoss()
        else:
            outputs = np.array(outputs)[:, None]
            loss_func = MSELoss()

        # Determine the scaling transform
        mean_y = torch.from_numpy(np.nanmean(outputs, axis=0)).float()
        std_y = torch.from_numpy(np.nanstd(outputs, axis=0)).float()
        transform = MeanScaler(mean=mean_y, var=std_y)
        outputs = torch.from_numpy(outputs).float()

        # Prepare for early stopping
        if patience is None:
            patience = max(1, max_epochs // 8)
        early_stop = EarlyStopping('val_loss', patience=patience, verbose=verbose)

        # Make the data loader
        train_data, val_data = train_test_split(list(zip(inputs, outputs)), test_size=validation_size)
        data_module = RedoxDataModule(
            train_data=train_data,
            valid_data=val_data,
            batch_size=batch_size,
            transforms=[transform],
            num_workers=num_workers,
        )

        # Run the training in a temporary directory
        with TemporaryDirectory() as tmpdir:
            model, kwargs = model_msg
            kwargs['output_dim'] = 1 if lower_fidelities is None else outputs.shape[1]
            task = RedoxTask(
                model,
                kwargs,
                lr=learning_rate,
                weight_decay=0.0,
                loss_func=loss_func
            )
            trainer = pl.Trainer(max_epochs=max_epochs,
                                 default_root_dir=tmpdir,
                                 enable_checkpointing=False,
                                 enable_progress_bar=verbose,
                                 enable_model_summary=verbose,
                                 accelerator="auto",
                                 callbacks=[early_stop])
            trainer.fit(task, datamodule=data_module)

            # Return the model as a serialized object
            fp = BytesIO()
            torch.save([task.encoder, transform], fp)
            return fp.getvalue()

    def update(self, model: ModelObjectType, update_msg: bytes) -> ModelObjectType:
        return model[0], update_msg

    def score(self,
              model_msg: bytes,
              inputs: list,
              lower_fidelities: np.ndarray | None = None,
              batch_size: int = 32,
              num_workers: int = 4,
              device: str | None = None) -> np.ndarray:
        # Unpack the model
        model, transform = torch.load(BytesIO(model_msg), map_location='cpu')

        # Move the model to the desired device
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        # Make the data loader
        with_targets = zip_longest(inputs, '', fillvalue=None)
        data_module = RedoxData(with_targets)
        loader = PyGLoader(data_module, batch_size=batch_size, num_workers=num_workers)

        # Run over all data
        with torch.no_grad():
            outputs = []
            for batch in loader:
                batch = batch.to(device)
                pred_y_unscaled = model(batch).cpu()
                pred_y = transform.inverse_transform(pred_y_unscaled)
                outputs.append(pred_y.detach().cpu().numpy())
        outputs = np.squeeze(np.concatenate(outputs, axis=0))

        # If needed, compute the delta
        if outputs.ndim == 1:
            return outputs

        known_deltas = compute_deltas(lower_fidelities)
        is_known = np.isfinite(known_deltas)
        outputs[:, :-1] = np.where(is_known, known_deltas, outputs[:, :-1])
        return outputs.sum(axis=1)
