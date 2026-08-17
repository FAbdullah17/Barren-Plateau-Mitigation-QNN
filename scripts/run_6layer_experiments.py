#!/usr/bin/env python
"""
Run all 6-layer experiments (3 approaches × seed-triple indices).

Executes baseline, layerwise, and local cost training at 6-layer depth across
all seed-triple indices . Each run derives its
(data, init, training) seed triple from the config's ``seeds.base_seed``.

Usage:
    python scripts/run_4layer_experiments.py
    python scripts/run_6layer_experiments.py
"""

import subprocess
import sys
import argparse
import time
from pathlib import Path

import yaml


EXPERIMENTS = [
    ('baseline', 'configs/baseline_6layer.yaml'),
    ('layerwise', 'configs/layerwise_6layer.yaml'),
    ('local_cost', 'configs/local_cost_6layer.yaml'),
]


def load_seed_triples(config_path: str) -> int:
    """Read ``seeds.seed_triples`` from a config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return int(config['seeds']['seed_triples'])


def run_experiment(approach: str, config: str, seed_index: int) -> bool:
    """Run a single experiment (one seed-triple index)."""
    script_map = {
        'baseline': 'experiments/run_baseline.py',
        'layerwise': 'experiments/run_layerwise.py',
        'local_cost': 'experiments/run_local_cost.py'
    }
    cmd = [
        sys.executable, script_map[approach], config,
        '--seed-index', str(seed_index),
    ]
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False


def main():
    default_indices = list(range(load_seed_triples(EXPERIMENTS[0][1])))
    parser = argparse.ArgumentParser(description='Run all 6-layer experiments')
    parser.add_argument(
        '--seed-indices', type=int, nargs='+', default=default_indices,
        help=f'Seed-triple indices to run (default: {default_indices})',
    )
    parser.add_argument(
        '--approach', type=str, choices=['baseline', 'layerwise', 'local_cost'],
        help='Run only a specific approach',
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would run without executing')
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.approach:
        experiments = [(a, c) for a, c in EXPERIMENTS if a == args.approach]

    total = len(experiments) * len(args.seed_indices)

    print("=" * 70)
    print("6-LAYER EXPERIMENTS")
    print("=" * 70)
    print(f"Approaches: {len(experiments)}")
    print(f"Seed indices: {args.seed_indices}")
    print(f"Total experiments: {total}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Would execute:")
        for approach, config in experiments:
            for index in args.seed_indices:
                print(f"  python experiments/run_{approach}.py {config} --seed-index {index}")
        return

    start_time = time.time()
    completed = 0
    failed = 0

    for approach, config in experiments:
        print(f"\n{'=' * 70}")
        print(f"APPROACH: {approach.upper()}")
        print(f"{'=' * 70}")

        for index in args.seed_indices:
            completed += 1
            print(f"\n[{completed}/{total}] Running {approach} with seed-index {index}...")

            success = run_experiment(approach, config, index)

            if success:
                print(f"✓ {approach} seed-index {index} completed")
            else:
                failed += 1
                print(f"✗ {approach} seed-index {index} FAILED")

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("6-LAYER EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"Completed: {completed - failed}/{total}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()