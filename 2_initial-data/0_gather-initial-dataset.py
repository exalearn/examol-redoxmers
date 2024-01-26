"""Compute an initial dataset for various recipes"""
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from queue import LifoQueue
from platform import node
from pathlib import Path
import logging
import gzip
import sys

import numpy as np
from rdkit import RDLogger
from proxystore.connectors.redis import RedisConnector
from proxystore.store import Store, register_store
from colmena.thinker import BaseThinker, ResourceCounter, task_submitter, result_processor
from colmena.task_server.parsl import ParslTaskServer
from colmena.queue import PipeQueues
from colmena.models import Result

from examol.store.db.memory import InMemoryStore
from examol.store.recipes import RedoxEnergy, SolvationEnergy

import configs

RDLogger.DisableLog('rdApp.*')


# Make the steering application
class BruteForceThinker(BaseThinker):
    """Evaluate all molecules in a user-provided list

    Args:
        queues: Queues used to communicate with task server
        args: Arguments provided to the run script
        slots: Number of molecules to run concurrently
        database: All known information about each molecule
        recipes: List of recipes to be computed
        optimization_path: Path in which to store optimization steps
    """

    def __init__(self,
                 queues: PipeQueues,
                 args: Namespace,
                 slots: int,
                 molecules: list[str],
                 database: InMemoryStore,
                 recipes: list[RedoxEnergy],
                 optimization_path: Path | None):
        super().__init__(queues, ResourceCounter(slots))
        self.optimization_path = optimization_path
        self.database = database
        self.recipes = recipes
        self.args = args

        # Determine where to store the task records
        run_name = self.database.path.name[:-8]
        self.record_path = self.database.path.parent / f'{run_name}-results.json.gz'

        # Output handles
        self.record_fp = gzip.open(self.record_path, 'at')
        self.optimization_fp = None
        if self.optimization_path is not None:
            self.optimization_fp = gzip.open(self.optimization_path, 'at')

        # State tracking
        self.molecule_queue: LifoQueue[str] = LifoQueue()  # List of molecules which are yet to be computed
        self.ongoing_tasks: dict[str, int] = defaultdict(int)  # How many tasks ongoing for each molecule
        self.failures: set[str] = set()  # Molecules which have had one failure and we should not re-add to queue
        self.success_count = 0

        # Push the molecules into the queue
        for mol in molecules:
            self.molecule_queue.put(mol)

    @task_submitter()
    def submit_task(self):
        """Submit tasks for a new molecule"""

        # Loop until we either find new work or run out of molecules
        while not self.molecule_queue.empty():
            # Get the next record
            key = self.molecule_queue.get()
            my_record = self.database.get_or_make_record(key)

            # See if there is any new work to do
            for recipe in self.recipes:
                try:
                    next_calculations = recipe.suggest_computations(my_record)
                    if len(next_calculations) > 0:
                        self.logger.info(f'Submitting tasks for {my_record.key} recipe {recipe.name}@{recipe.level}')
                        # Submit them
                        for request in next_calculations:
                            method = 'optimize_structure' if request.optimize else 'compute_energy'
                            self.queues.send_inputs(
                                my_record.key, request.xyz, request.config_name, request.charge, request.solvent,
                                method=method,
                                task_info={'key': my_record.key, 'recipe': recipe.name, 'level': recipe.level}
                            )
                            self.ongoing_tasks[my_record.key] += 1

                        # Finished!
                        self.logger.info(f'Submitted tasks from {my_record.key} for {recipe.name}@{recipe.level}. '
                                         f'{self.molecule_queue.qsize()} in queue. {len(self.ongoing_tasks)} under way')
                        return

                    # Compute the property
                    recipe.update_record(my_record)
                except ValueError as e:
                    self.logger.warning(f'{my_record.key} failed for {recipe.name}@{recipe.level}. Error: {e}')
                    if self.args.halt_on_error:
                        raise ValueError(f'Failed to submit new tasks for {my_record.key}')

        # If there are neither molecules no ongoing tasks, then we are done
        if len(self.ongoing_tasks) == 0:
            self.done.set()

    @result_processor()
    def store_result(self, result: Result):
        """Store the result from a computation"""

        # Get the associated record
        my_record = self.database[result.task_info['key']]
        self.ongoing_tasks[my_record.key] -= 1

        # Save the result to disk
        print(result.json(exclude={'value', 'inputs'}), file=self.record_fp)

        # Update the record
        if not result.success:
            self.logger.warning(f'Computation failed for {my_record.key}. Error: {result.failure_info}')
            if args.halt_on_error:
                raise ValueError(f'Computation failed. Halting everything')
            self.failures.add(my_record.key)
        else:
            if result.method == 'optimize_structure':
                sim_result, steps, metadata = result.value
                my_record.add_energies(sim_result, steps)

                # Save the steps to disk
                if self.optimization_fp is not None:
                    for step in steps:
                        print(step.json(), file=self.optimization_fp)
            else:
                sim_result, metadata = result.value
                my_record.add_energies(sim_result)

        # Trigger an update to the database
        self.database.update_record(my_record)

        # Check if we need more work for this molecule if there was at least one success
        self.logger.info(f'Stored record for {my_record.key}. {self.ongoing_tasks[my_record.key]} tasks remaining')
        if self.ongoing_tasks[my_record.key] == 0:
            # Only re-submit molecule if there were no failures
            if my_record.key not in self.failures:
                self.molecule_queue.put(my_record.identifier.smiles)
            self.ongoing_tasks.pop(my_record.key)  # Mark that this molecule is done
            self.rec.release()  # Let the next molecule start
            self.success_count += 1


if __name__ == "__main__":
    # Make the argument parser
    parser = ArgumentParser()
    parser.add_argument('search_space', help='Path to the SMI file containing strings to be run')
    parser.add_argument('--config-function', default='make_local_config', help='Name of the configuration function from `config.py`')
    parser.add_argument('--num-to-run', default=None, help='Maximum number of molecules to run', type=int)
    parser.add_argument('--halt-on-error', action='store_true', help='Halt the workflow if a single task fails')
    parser.add_argument('--write-frequency', default=5, help='Minimum frequency of saving database', type=float)
    args = parser.parse_args()

    # Make a logger
    my_logger = logging.getLogger('main')
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler('runtime.log', 'a')]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        for logger in [logging.getLogger('examol'), my_logger]:
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    # Load in the molecules
    search_path = Path(args.search_space)
    my_logger.info(f'Loading molecules from {search_path}')
    with search_path.open() as fp:
        molecules = [line.strip() for line in fp]
    my_logger.info(f'Loaded {len(molecules)} molecules to screen')

    # Downselect to a certain number of molecules
    rng = np.random.default_rng(seed=1)
    rng.shuffle(molecules)
    if args.num_to_run is not None:
        molecules = molecules[:args.num_to_run]
        my_logger.info(f'Down selected to {len(molecules)} molecules')

    # Get the path to the dataset
    dataset_path = Path('datasets') / f'{search_path.name[:-4]}.json.gz'
    records_path = dataset_path.parent / f'{search_path.name[:-4]}-simulation-records.json.gz'
    dataset = InMemoryStore(dataset_path, args.write_frequency)

    # Get the right computational environment
    config_fn = getattr(configs, args.config_function)
    config, sim, n_slots, energy_configs = config_fn()
    my_logger.info(f'Loaded configuration function "{args.config_function}"')

    # Get the recipes we should run
    recipes = []
    solvents = ['acn']
    for energy_level in energy_configs:
        for charge in [-1]:
            for solvent in [None] + solvents:
                recipes.extend([
   #                 RedoxEnergy(energy_config=energy_level, vertical=True, charge=charge, solvent=solvent),
                    RedoxEnergy(energy_config=energy_level, vertical=False, charge=charge, solvent=solvent),
                ])
        for solvent in solvents:
            recipes.append(SolvationEnergy(config_name=energy_level, solvent=solvent))
    my_logger.info(f'Assembled a list of {len(recipes)} recipes to compute')

    # Create the queues which will connect task server and thinker
    queues = PipeQueues()

    # Create the task server
    task_server = ParslTaskServer(queues=queues, methods=[sim.compute_energy, sim.optimize_structure], config=config)
    my_logger.info('Created task server')

    # Create the thinker
    thinker = BruteForceThinker(queues, args, n_slots, molecules, dataset, recipes, records_path)
    for handler in handlers:
        thinker.logger.addHandler(handler)
    thinker.logger.setLevel(logging.INFO)

    # Run the script
    try:
        with dataset:
            task_server.start()
            thinker.run()
    finally:
        my_logger.info(f'Closing logs')
        thinker.record_fp.close()
        if thinker.optimization_fp is not None:
            thinker.optimization_fp.close()
        queues.send_kill_signal()
    task_server.join()

    # Print whether we completed many molecules
    my_logger.info(f'Added {thinker.success_count} new computations.')

    # Close the proxystore
    store.close()
    my_logger.info('Done!')
