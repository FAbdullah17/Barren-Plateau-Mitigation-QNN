#!/usr/bin/env python
"""
Batch experiment runner for Barren-Plateau-Mitigation-QNN project.
Runs multiple seed-triple indices for a given approach and configuration
Seed triples are derived from the config's ``seeds.base_seed``.

Usage:
    python scripts/run_batch.py baseline configs/baseline_8layer.yaml
    python scripts/run_batch.py baseline configs/baseline_8layer.yaml --seed-indices 0 1 2
"""

import subprocess
import sys
import argparse
from pathlib import Path
import time

import yaml


def load_seed_triples(config_path: str) -> int:
    """Read ``seeds.seed_triples`` from a config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return int(config['seeds']['seed_triples'])


def run_experiment(approach: str, config_path: str, seed_index: int) -> bool:
    """Run a single experiment (one seed-triple index)."""
    script_map = {
        'baseline': 'experiments/run_baseline.py',
        'layerwise': 'experiments/run_layerwise.py',
        'local_cost': 'experiments/run_local_cost.py'
    }

    if approach not in script_map:
        print(f"ERROR: Unknown approach '{approach}'")
        print(f"Valid approaches: {list(script_map.keys())}")
        return False

    script = script_map[approach]
    cmd = [sys.executable, script, config_path, '--seed-index', str(seed_index)]

    print(f"\n{'=' * 70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 70}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    default_indices = list(range(load_seed_triples('configs/baseline_8layer.yaml')))
    parser = argparse.ArgumentParser(description='Run batch experiments over seed-triple indices')
    parser.add_argument('approach', type=str, choices=['baseline', 'layerwise', 'local_cost'],
                        help='Experiment approach')
    parser.add_argument('config', type=str, help='Path to config YAML file')
    parser.add_argument(
        '--seed-indices', type=int, nargs='+', default=None,
        help='Seed-triple indices to run (default: all indices in seeds.seed_triples)',
    )
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Continue running even if one seed fails')
    args = parser.parse_args()

    if args.seed_indices is None:
        n_triples = load_seed_triples(args.config)
        args.seed_indices = list(range(n_triples))

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"BATCH EXPERIMENT RUNNER")
    print(f"{'=' * 70}")
    print(f"Approach: {args.approach}")
    print(f"Config: {args.config}")
    print(f"Seed indices: {args.seed_indices}")
    print(f"Total experiments: {len(args.seed_indices)}")
    print(f"{'=' * 70}")

    start_time = time.time()
    successful = 0
    failed = 0

    for i, seed_index in enumerate(args.seed_indices, 1):
        print(f"\n[{i}/{len(args.seed_indices)}] Running seed-index {seed_index}")

        success = run_experiment(args.approach, args.config, seed_index)

        if success:
            successful += 1
            print(f"✓ Seed-index {seed_index} completed successfully")
        else:
            failed += 1
            print(f"✗ Seed-index {seed_index} FAILED")

            if not args.continue_on_error:
                print("\nAborting batch run due to failure.")
                print("Use --continue-on-error to continue despite failures.")
                break

    total_time = time.time() - start_time

    print(f"\n{'=' * 70}")
    print(f"BATCH RUN COMPLETE")
    print(f"{'=' * 70}")
    print(f"Successful: {successful}/{len(args.seed_indices)}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"{'=' * 70}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()