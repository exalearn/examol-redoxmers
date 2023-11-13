# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from argparse import ArgumentParser, Action, Namespace
from functools import partial
from multiprocessing import Pool
from typing import Sequence, Any
from pathlib import Path
import gzip
import json

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split

from examol.store.models import MoleculeRecord, MissingData
from exaredox.score.pytorch import models, get_graph_information
from exaredox.score.pytorch.task import RedoxTask
from exaredox.score.pytorch.data import RedoxDataModule
from exaredox.score.pytorch.transforms import MeanScaler


class ModelKwargParser(Action):
    def __call__(
            self,
            parser: ArgumentParser,
            namespace: Namespace,
            values: str | Sequence[Any] | None,
            option_string: str | None = ...,
    ) -> None:
        setattr(namespace, self.dest, dict())
        if values:
            for value in values:
                key, value = value.split("=")
                # for numeric data, type cast into float or int
                if value.isnumeric():
                    value = float(value)
                    value = int(value) if value.is_integer() else value
                getattr(namespace, self.dest)[key] = value


def _parse_line(line, property_name, method_name):
    """Parse a line from the training set into a form readable by PyTorch"""
    record: MoleculeRecord = MoleculeRecord.parse_raw(line)
    try:
        conf, _ = record.find_lowest_conformer(config_name='mmff', charge=0, solvent=None)
    except MissingData:
        return
    target = record.properties[property_name][method_name]
    return (get_graph_information(conf.xyz), None), target


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Example script for training redox models with a single process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12156906,
        help="Random seed passed to `pl.seed_everything`.",
    )
    parser.add_argument(
        "-t",
        "--data-path",
        default='../datasets/mdf-mos',
        type=str,
        help="Path to the subset of training data being used for training",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=5e-4,
        type=float,
        help="Learning rate passed into the AdamW optimizer.",
    )
    parser.add_argument(
        "-w",
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay regularization passed into the AdamW optimizer.",
    )
    parser.add_argument("-k", "--model-kwargs", nargs="*", action=ModelKwargParser)
    parser.add_argument(
        "-n",
        "--num-loaders",
        type=int,
        default=4,
        help="Number of parallel data loader workers.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        help="Number of training samples per batch.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="EGNN",
        help="Class name of model, retrieved from `redox_models.models`.",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "-f",
        "--fast-runs",
        type=int,
        default=None,
        help="If an integer is passed, this will run the specified number of batches"
             "as a `fast_dev_run` that skips checkpointing and logging for debugging.",
    )
    parser.add_argument(
        "-tp",
        "--target-property",
        type=str,
        default="oxidation_potential",
        help="Name of the property to use as target labels.",
    )
    parser.add_argument(
        "-tm",
        "--target-method",
        type=str,
        default="xtb-vertical",
        help="Name of the method used to calculate target labels.",
    )

    args = parser.parse_args()

    # Determine the path to data directory from the data path, target method and property
    data_path = Path(args.data_path) / f'{args.target_property}-{args.target_method}'

    # Load in the training data, using the initial geometry as an input
    def _read_data(path: Path) -> list[tuple[tuple[dict, float | None], float]]:
        with gzip.open(path, 'rt') as fp:
            fun = partial(_parse_line, property_name=args.target_property, method_name=args.target_method)
            with Pool() as pool:
                return [x for x in pool.imap(fun, fp, chunksize=1024) if x is not None]


    train_data = _read_data(data_path / 'train.json.gz')
    test_data = _read_data(data_path / 'test.json.gz')

    # Determine the mean and standard deviation from the
    targets = [x[1] for x in train_data]
    label_mean = float(np.mean(targets))
    label_stddev = float(np.std(targets))
    scaler = MeanScaler(label_mean, var=label_stddev)

    # Split validation data off of train data
    train_data, valid_data = train_test_split(train_data, test_size=0.1)

    # sets the random seed for torch, numpy, python, etc.
    pl.seed_everything(args.seed)

    # instantiate data module that orchestrates the splits and loading
    dm = RedoxDataModule(
        train_data=train_data,
        valid_data=valid_data,
        predict_data=test_data,
        transforms=[scaler],
        num_workers=args.num_loaders,
        batch_size=args.batch_size,
    )

    # extract out the class reference from models
    encoder = getattr(models, args.model)

    # construct the task
    model_kwargs = dict(
        output_dim=1
    )
    if args.model_kwargs:
        model_kwargs.update(args.model_kwargs)
    task = RedoxTask(
        encoder,
        model_kwargs,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # instantiate trainer, specifying how training is done
    logger = CSVLogger('.')
    trainer = pl.Trainer(max_epochs=args.epochs, fast_dev_run=args.fast_runs, accelerator='gpu', logger=logger, log_every_n_steps=1000)

    # run the training loop
    trainer.fit(task, datamodule=dm)

    # Run the prediction tasks
    pred_y = trainer.predict(task, datamodule=dm)
    pred_y = scaler.inverse_transform(torch.concat(pred_y))
    pred_y = np.squeeze(pred_y.detach().cpu().numpy())

    # Save run data to disk
    true_y = [x[1] for x in test_data]
    pd.DataFrame({'true': true_y, 'pred': pred_y}).to_csv(Path(logger.log_dir) / 'predictions.csv.gz', index=False)

    # Save the arguments
    with open(Path(logger.log_dir) / 'params.json', 'w') as fp:
        json.dump(args.__dict__, fp)
