#!/usr/bin/env python
"""
Run the notebook's FQE sweep across several worker processes.

The FQE runs are independent of one another, and the model is small enough that
a single GPU is nowhere near saturated by one of them, so running a few workers
over the same GPU is close to a linear speed-up.

    # 4 workers over one GPU
    for i in 0 1 2 3; do python run_fqe_sweep.py --shard $i --num-shards 4 & done
    wait

    # or one worker per GPU
    CUDA_VISIBLE_DEVICES=0 python run_fqe_sweep.py --shard 0 --num-shards 2 &
    CUDA_VISIBLE_DEVICES=1 python run_fqe_sweep.py --shard 1 --num-shards 2 &
    wait

Each worker executes the notebook's own cells - there is no second copy of the
training code anywhere - but only the runs it owns (run_index % num_shards ==
shard), and writes to its own results file, results/fqe_v_s0.shard<i>.npz.

Afterwards, run the notebook's FQE cell and then its Figure 4 cell: every run
will already be recorded, so nothing is retrained; the cell just merges the
shards, prints the bootstrapped IQM table and draws the shaded figure.

Requires MULTI_SEED = True in the notebook's configuration cell.
"""
import argparse
import json
import os
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--shard', type=int, default=0, help='This worker\'s index (0-based).')
parser.add_argument('--num-shards', type=int, default=1, help='Total number of workers.')
parser.add_argument('--notebook', default='tutorial_notebook.ipynb', help='Notebook to take the code from.')
args = parser.parse_args()

if not 0 <= args.shard < args.num_shards:
    parser.error(f'--shard must be in [0, {args.num_shards}).')

# Workers are headless and must not try to open a plotting window. Set before
# anything imports matplotlib.
os.environ['MPLBACKEND'] = 'Agg'
os.environ['FQE_SHARD_INDEX'] = str(args.shard)
os.environ['FQE_NUM_SHARDS'] = str(args.num_shards)
os.environ['FQE_WORKER'] = '1'

# The cells this worker needs, in order, identified by a distinctive substring
# rather than an index so that editing the notebook does not silently break it.
CELL_ANCHORS = [
    'IMPORT STATEMENTS',                              # imports, DEVICE, console_print
    'MULTI-SEED / CONFIDENCE-INTERVAL CONFIGURATION',  # seeds, directories, fqe_seed_for
    'all_data = pl.scan_parquet',                     # lazy frame used below
    'val_dataset = ExampleTrainingDataset',           # validation split
    'train_dataset = ExampleTrainingDataset',         # training split + dataloaders
    'output_doses = np.array',                        # dose categories
    'def get_dose_probs_from_logits',                 # ordinal helpers
    'def magni_risk_index(',                          # reward function
    'v_s0_reference = (',                             # ground truth return
    'fqe_results = ResultsStore',                     # the FQE sweep itself
]

notebook_path = os.path.abspath(args.notebook)
repo_root = os.path.dirname(notebook_path)
sys.path.insert(0, repo_root)
os.chdir(repo_root)

with open(notebook_path) as f:
    sources = [''.join(cell['source']) for cell in json.load(f)['cells'] if cell['cell_type'] == 'code']


def find_cell(anchor):
    matches = [source for source in sources if anchor in source]
    if len(matches) != 1:
        raise SystemExit(
            f'Expected exactly one code cell containing {anchor!r}, found {len(matches)}. '
            f'The notebook has diverged from this script - update CELL_ANCHORS.'
        )
    return matches[0]


namespace = {'__name__': '__main__'}

print(f'[worker {args.shard + 1}/{args.num_shards}] running cells from {notebook_path}')
for anchor in CELL_ANCHORS:
    source = find_cell(anchor)

    if anchor == 'MULTI-SEED / CONFIDENCE-INTERVAL CONFIGURATION':
        exec(compile(source, f'<{anchor}>', 'exec'), namespace)
        if not namespace['MULTI_SEED']:
            raise SystemExit(
                'MULTI_SEED is False in the notebook, so there is only one FQE run and '
                'nothing to parallelise. Set MULTI_SEED = True and re-run.'
            )
        # Nothing is watching, so do not spend time rendering live plots.
        namespace['PLOT_EVERY'] = 10 ** 9
        continue

    if anchor == 'train_dataset = ExampleTrainingDataset':
        exec(compile(source, f'<{anchor}>', 'exec'), namespace)
        # The notebook picks these up from the earlier training cells, which a
        # worker does not run.
        namespace['n_train'] = len(namespace['train_dataloader'])
        namespace['n_val'] = len(namespace['val_dataloader'])
        continue

    exec(compile(source, f'<{anchor}>', 'exec'), namespace)

print(f'[worker {args.shard + 1}/{args.num_shards}] done.')
