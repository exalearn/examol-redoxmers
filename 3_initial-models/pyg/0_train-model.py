# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from argparse import ArgumentParser, Action, Namespace
from typing import Sequence, Any
from pathlib import Path
from hashlib import md5
import gzip
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from lightning import pytorch as pl
from sklearn import metrics

from examol.score.utils.multifi import collect_outputs
from examol.store.models import MoleculeRecord
from examol.store.recipes import RedoxEnergy
from exaredox.score.pytorch import RedoxModelsScorer


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
    parser.add_argument("-k", "--model-kwargs", nargs="*", action=ModelKwargParser, default={})
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
        help="Class name of model, retrieved from `exaredox.score.pytorch`",
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
    parser.add_argument(
        "--lower-levels",
        type=str,
        nargs='*',
        help="Lower levels of fidelity used to enhance training",
    )

    args = parser.parse_args()

    # Assemble the recipes and lower-fidelity data
    top_recipe = RedoxEnergy.from_name(args.target_property, args.target_method)
    all_recipes: list[RedoxEnergy] = [
                                         RedoxEnergy.from_name(top_recipe.name, level) for level in args.lower_levels or []
                                     ] + [top_recipe]
    print(f'Learning {top_recipe.name} at levels: {", ".join(x.level for x in all_recipes)}')

    # Load in the training data, using the initial geometry as an input
    # Determine the path to data directory from the data path, and lowest-level method and property
    lowest_recipe = all_recipes[0]
    data_path = Path(args.data_path) / f'{lowest_recipe.name}-{lowest_recipe.level}'

    def _read_data(path: Path) -> list[MoleculeRecord]:
        with gzip.open(path, 'rt') as fp:
            return [MoleculeRecord.parse_raw(l) for l in tqdm(fp, desc=path.name)]


    train_data = _read_data(data_path / 'train.json.gz')
    test_data = _read_data(data_path / 'test.json.gz')
    data_hash = (data_path / 'dataset.md5').read_text().strip()

    #  Make a run directory
    run_settings = args.__dict__.copy()
    run_settings['train_counts'] = len(train_data)
    run_settings['name'] = top_recipe.name
    run_settings['level'] = top_recipe.level
    run_settings['data_hash'] = data_hash
    settings_hash = md5(json.dumps(args.__dict__).encode()).hexdigest()[-8:]
    run_dir = Path(f'runs/model={args.model}-prop={args.target_property}_{args.target_method}-levels={len(all_recipes)}-hash={settings_hash}')
    if (run_dir / 'test_results.csv').exists():
        raise ValueError('Run already done')
    run_dir.mkdir(exist_ok=True, parents=True)
    (run_dir / 'params.json').write_text(json.dumps(run_settings))

    # sets the random seed for torch, numpy, python, etc.
    pl.seed_everything(args.seed)

    # Run training
    scorer = RedoxModelsScorer()
    train_inputs = scorer.transform_inputs(train_data)
    train_outputs = scorer.transform_outputs(train_data, top_recipe)
    lower_fidelities = None
    if len(all_recipes) > 1:
        lower_fidelities = collect_outputs(train_data, all_recipes[:-1])
    del train_data
    model_msg = scorer.retrain((args.model, args.model_kwargs), train_inputs, train_outputs,
                               max_epochs=args.epochs, batch_size=args.batch_size,
                               num_workers=args.num_loaders,
                               lower_fidelities=lower_fidelities,
                               verbose=True)

    # Run the prediction tasks
    test_data = [record for record in test_data if top_recipe.level in record.properties[top_recipe.name]]
    test_inputs = scorer.transform_inputs(test_data)
    test_outputs = np.array([record.properties[top_recipe.name].pop(top_recipe.level) for record in test_data])
    summary = {}
    all_preds = {'smiles': [r.identifier.smiles for r in test_data],
                 'true': test_outputs.squeeze()}
    for level_id, recipe in enumerate(all_recipes[::-1]):
        # Remove that level
        for record in test_data:
            record.properties[recipe.name].pop(recipe.level, None)
        if len(all_recipes) > 1:
            lower_fidelities = collect_outputs(test_data, all_recipes[:-1])
        print(f'Running inference with ')

        # Create the inputs
        test_preds = scorer.score(model_msg, test_inputs, lower_fidelities=lower_fidelities)

        # Append the summary and score
        level_tag = f'level_{len(all_recipes) - level_id - 1}'
        summary.update(dict(
            (f'{level_tag}_{f.__name__}', f(test_outputs, test_preds)) for f in
            [metrics.mean_absolute_error, metrics.r2_score, metrics.mean_squared_error]
        ))
        all_preds[f'{level_tag}-pred'] = test_preds.squeeze()
    # Save results
    (run_dir / 'test_summary.json').write_text(json.dumps(summary, indent=2))
    pd.DataFrame(all_preds).to_csv(run_dir / 'test_results.csv', index=False)

    # Save results
    (run_dir / 'test_summary.json').write_text(json.dumps(summary, indent=2))
    pd.DataFrame(all_preds).to_csv(run_dir / 'test_results.csv', index=False)
