#!/usr/bin/env python
"""
Validate experiment results for completeness and correctness.

Checks each experiment directory contains the canonical files (metrics.json,
training_history.png) and that metrics.json carries all required fields with
sane values.

Usage:
    python scripts/validate_results.py results/
    python scripts/validate_results.py results/baseline/depth_4/seed_0/
"""

import json
import argparse
from pathlib import Path
import sys


REQUIRED_FIELDS = [
    'config',
    'data_seed',
    'init_seed',
    'training_seed',
    'seed_index',
    'test_acc',
    'test_loss',
    'training_time_seconds',
    'total_updates',
    'n_parameters',
    'training_diagnostic',
    'history',
    'pca_info',
]

REQUIRED_FILES = [
    'metrics.json',
    'training_history.png',
]

# Nested fields that must be present inside the diagnostic objects.
TRAINING_DIAGNOSTIC_FIELDS = [
    'n_parameters', 'mean_param_grad_variance', 'std_param_grad_variance',
    'mean_abs_grad', 'max_abs_grad', 'trajectory',
]
TRAJECTORY_FIELDS = ['step', 'mean_param_grad_variance']
HISTORY_FIELDS = ['step', 'train_loss', 'train_acc', 'val_loss', 'val_acc']


def _validate_metrics(metrics: dict) -> str:
    """Return an empty string if metrics are valid, else an error message."""
    missing = [f for f in REQUIRED_FIELDS if f not in metrics]
    if missing:
        return f"Missing fields: {missing}"

    test_acc = metrics['test_acc']
    if not (0.0 <= test_acc <= 1.0):
        return f"Invalid test_acc: {test_acc} (must be in [0, 1])"

    if metrics['training_time_seconds'] < 0:
        return f"Invalid training_time_seconds: {metrics['training_time_seconds']}"

    if metrics['total_updates'] < 1:
        return f"Invalid total_updates: {metrics['total_updates']}"

    if metrics['n_parameters'] < 1:
        return f"Invalid n_parameters: {metrics['n_parameters']}"

    diagnostic = metrics['training_diagnostic']
    missing = [f for f in TRAINING_DIAGNOSTIC_FIELDS if f not in diagnostic]
    if missing:
        return f"training_diagnostic missing fields: {missing}"
    missing = [f for f in TRAJECTORY_FIELDS if f not in diagnostic['trajectory']]
    if missing:
        return f"training_diagnostic.trajectory missing fields: {missing}"
    if not diagnostic['trajectory']['step']:
        return "training_diagnostic.trajectory.step is empty"

    history = metrics['history']
    missing = [f for f in HISTORY_FIELDS if f not in history]
    if missing:
        return f"history missing fields: {missing}"
    if len(history['step']) != len(history['train_loss']):
        return "history step/train_loss length mismatch"

    if metrics['pca_info'].get('n_components', 0) < 1:
        return "pca_info.n_components invalid"

    return ""


def validate_experiment(result_dir: Path) -> tuple:
    """
    Check if a single experiment completed successfully.

    Returns:
        (is_valid, message)
    """
    result_dir = Path(result_dir)

    for filename in REQUIRED_FILES:
        filepath = result_dir / filename
        if not filepath.exists():
            return False, f"Missing file: {filename}"

    metrics_path = result_dir / "metrics.json"
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in metrics.json: {e}"
    except OSError as e:
        return False, f"Cannot read metrics.json: {e}"

    message = _validate_metrics(metrics)
    if message:
        return False, message

    return True, "Valid"


def find_experiment_dirs(base_dir: Path) -> list:
    """Find all experiment result directories."""
    base_dir = Path(base_dir)
    experiments = []

    if (base_dir / 'metrics.json').exists():
        return [base_dir]

    for metrics_file in base_dir.rglob('metrics.json'):
        experiments.append(metrics_file.parent)

    return experiments


def main():
    parser = argparse.ArgumentParser(description='Validate experiment results')
    parser.add_argument('path', type=str, help='Path to results directory or specific experiment')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show details for each experiment')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()

    results_path = Path(args.path)
    if not results_path.exists():
        print(f"ERROR: Path not found: {args.path}")
        sys.exit(1)

    experiments = find_experiment_dirs(results_path)

    if not experiments:
        print(f"No experiments found in: {args.path}")
        sys.exit(1)

    print(f"\nValidating {len(experiments)} experiment(s)...\n")

    results = []
    valid_count = 0
    invalid_count = 0

    for exp_dir in sorted(experiments):
        is_valid, message = validate_experiment(exp_dir)

        relative_path = exp_dir.relative_to(results_path) if exp_dir != results_path else exp_dir.name

        result = {
            'path': str(relative_path),
            'valid': is_valid,
            'message': message
        }
        results.append(result)

        if is_valid:
            valid_count += 1
            if args.verbose:
                print(f"✓ {relative_path}")
        else:
            invalid_count += 1
            print(f"✗ {relative_path}: {message}")

    print(f"\n{'='*50}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Valid: {valid_count}")
    print(f"Invalid: {invalid_count}")
    print(f"{'='*50}")

    if args.json:
        print("\nJSON Output:")
        print(json.dumps(results, indent=2))

    sys.exit(0 if invalid_count == 0 else 1)


if __name__ == "__main__":
    main()