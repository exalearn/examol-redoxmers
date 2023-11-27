"""Train a model and save results into a directory structure"""
from examol.score.utils.multifi import collect_outputs
from examol.store.recipes import RedoxEnergy
from examol.score.nfp import NFPScorer, make_simple_network
from examol.store.models import MoleculeRecord
from sklearn import metrics
from argparse import ArgumentParser
from pathlib import Path
from hashlib import md5
from tqdm import tqdm
import pandas as pd
import numpy as np
import gzip
import json

if __name__ == "__main__":
    # Parse input arguments
    parser = ArgumentParser()
    parser.add_argument('--atom-features', default=32, help='How many features to use to describe each atom/bond', type=int)
    parser.add_argument('--message-steps', default=4, help='How many message-passing steps', type=int)
    parser.add_argument('--output-layers', default=(64, 32, 32), help='Number of hidden units in the output layers', nargs='*', type=int)
    parser.add_argument('--reduce-op', default='sum', help='Operation used to combine atom- to -molecule-level features')
    parser.add_argument('--atomwise', action='store_true', help='Whether to combine to molecule-level features before or after output layers')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite the previous model run')
    parser.add_argument('--num-epochs', type=int, default=4, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Number of records per batch')
    parser.add_argument('--lower-levels', nargs='*', help='Lower levels of fidelity to use in model training and inference')
    parser.add_argument('property', choices=['ea', 'ip'], help='Name of the property to assess')
    parser.add_argument('level', help='Accuracy level of the property to predict')

    args = parser.parse_args()

    # Load in the training data
    prop: str = {
        'ea': 'reduction_potential',
        'ip': 'oxidation_potential',
    }[args.property]
    top_recipe = RedoxEnergy.from_name(prop, args.level)
    print(f'Training a model for {top_recipe.name}//{top_recipe.level}')

    all_recipes: list[RedoxEnergy] = [
                      RedoxEnergy.from_name(prop, level) for level in args.lower_levels
                  ] + [top_recipe]
    print(f'Learning using a total of {len(all_recipes)} recipes')

    data_level = args.level if len(args.lower_levels) == 0 else args.lower_levels[0]

    data_path = Path(f'../datasets/mdf-mos/{prop}-{data_level}')
    data_hash = (data_path / 'dataset.md5').read_text()
    with gzip.open(data_path / 'train.json.gz', 'rt') as fp:
        train_records = [MoleculeRecord.parse_raw(line) for line in tqdm(fp, desc='loading training...')]
    with gzip.open(data_path / 'test.json.gz', 'rt') as fp:
        test_records = [MoleculeRecord.parse_raw(line) for line in tqdm(fp, desc='loading test...')]
    print(f'Read from {data_path}. Found {len(train_records)} train, {len(test_records)} test records. Data hash: {data_hash}')

    # Count the number of records at each level of fidelity
    train_counts = [0] * len(all_recipes)
    for record in train_records:
        for i, recipe in enumerate(all_recipes):
            if recipe.level in record.properties[recipe.name]:
                train_counts[i] += 1
    print('Counts for energies at each level of fidelity')
    for level, count in zip(all_recipes, train_counts):
        print(f'  {level.level} - {count}')

    #  Make a run directory
    run_settings = args.__dict__.copy()
    run_settings.pop('overwrite')
    run_settings['train_counts'] = train_counts
    run_settings['name'] = top_recipe.name
    run_settings['level'] = top_recipe.level
    run_settings['data_hash'] = data_hash
    settings_hash = md5(json.dumps(args.__dict__).encode()).hexdigest()[-8:]
    run_dir = Path(f'runs/f={args.atom_features}-T={args.message_steps}-r={args.reduce_op}-atomwise={args.atomwise}-hash={settings_hash}')
    if (run_dir / 'test_results.csv').exists() and not args.overwrite:
        raise ValueError('Run already done')
    run_dir.mkdir(exist_ok=True, parents=True)
    (run_dir / 'params.json').write_text(json.dumps(run_settings))

    # Make the network
    model = make_simple_network(
        message_steps=args.message_steps,
        atom_features=args.atom_features,
        output_layers=args.output_layers,
        reduce_op=args.reduce_op,
        atomwise=args.atomwise,
        outputs=len(all_recipes)
    )
    print('Made the model')

    # Run the training
    scorer = NFPScorer(retrain_from_scratch=True)
    train_inputs = scorer.transform_inputs(train_records)
    train_outputs = scorer.transform_outputs(train_records, all_recipes[-1])
    lower_fidelities = None
    if len(all_recipes) > 0:
        lower_fidelities = collect_outputs(train_records, all_recipes[:-1])
    del train_records
    print(f'Converted data to input formats')

    model_msg = scorer.prepare_message(model, training=True)
    update_msg = scorer.retrain(model_msg, train_inputs, train_outputs, lower_fidelities=lower_fidelities,
                                verbose=True, num_epochs=args.num_epochs, batch_size=args.batch_size)
    scorer.update(model, update_msg)

    # Save the model and training log
    model.save(run_dir / 'model.keras')
    pd.DataFrame(update_msg[1]).to_csv(run_dir / 'log.csv', index=False)

    # Get the top level of fidelity from the test records to use for the target value
    test_records = [record for record in test_records if top_recipe.level in record.properties[top_recipe.name]]
    test_outputs = np.array([record.properties[top_recipe.name].pop(top_recipe.level) for record in test_records])
    print(f'{len(test_records)} of the test set have the top level')

    # Measure the performance on the hold-out set starting from different levels
    model_msg = scorer.prepare_message(model)
    summary = {}
    test_inputs = scorer.transform_inputs(test_records)
    all_preds = {'smiles': [r.identifier.smiles for r in test_records],
                 'true': test_outputs.squeeze()}
    for level_id, recipe in tqdm(enumerate(all_recipes[::-1]), desc='testing'):
        # Remove that level
        for record in test_records:
            record.properties[recipe.name].pop(recipe.level, None)
        if len(all_recipes) > 1:
            lower_fidelities = collect_outputs(test_records, all_recipes[:-1])

        # Create the inputs
        test_preds = scorer.score(model_msg, test_inputs, lower_fidelities=lower_fidelities)

        # Append the summary and score
        level_tag = f'level_{len(all_recipes) - level_id - 1}'
        summary.update(dict(
            (f'{level_tag }_{f.__name__}', f(test_outputs, test_preds)) for f in
            [metrics.mean_absolute_error, metrics.r2_score, metrics.mean_squared_error]
        ))
        all_preds[f'{level_tag}-pred'] = test_preds.squeeze()

    # Save results
    (run_dir / 'test_summary.json').write_text(json.dumps(summary, indent=2))
    pd.DataFrame(all_preds).to_csv(run_dir / 'test_results.csv', index=False)
