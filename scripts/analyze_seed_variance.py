#!/usr/bin/env python
"""
Analyze variance across different seed triples.

Summarizes per-seed test accuracy, test loss, training time, and the gradient
diagnostic ``mean_param_grad_variance`` across the ``seed_<N>`` results.

Usage:
    python scripts/analyze_seed_variance.py results/baseline/depth_8/
    python scripts/analyze_seed_variance.py results/baseline/depth_8/ --json
"""

import json
import argparse
import sys
from pathlib import Path

import numpy as np


# Fields required from each metrics.json for this analysis.
REQUIRED_FIELDS = [
    'seed_index', 'test_acc', 'test_loss', 'training_time_seconds',
    'n_parameters', 'training_diagnostic',
]


def load_metrics(result_dir: Path):
    """Load metrics.json from a result directory, or None if absent."""
    metrics_path = Path(result_dir) / 'metrics.json'
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        return json.load(f)


def analyze_variance(metrics_list: list) -> dict:
    """Per-metric mean/std/min/max statistics across seeds."""
    if not metrics_list:
        return None

    def _series(key, transform=None):
        if transform is None:
            transform = (lambda m: m[key])
        return np.asarray([transform(m) for m in metrics_list], dtype=float)

    test_acc = _series('test_acc')
    test_loss = _series('test_loss')
    train_time = _series('training_time_seconds')
    param_grad_var = _series(
        'training_diagnostic',
        lambda m: m['training_diagnostic']['mean_param_grad_variance'],
    )

    def _summary(values):
        return {
            'mean': float(values.mean()),
            'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            'min': float(values.min()),
            'max': float(values.max()),
            'values': [float(v) for v in values],
        }

    # Bootstrap 95% CI of the mean test accuracy (Monte-Carlo, seed=0).
    rng = np.random.default_rng(0)
    boot_idx = rng.integers(0, len(test_acc), size=(2000, len(test_acc)))
    ci = [float(np.percentile(test_acc[boot_idx].mean(axis=1), q)) for q in (2.5, 97.5)]

    stats = {
        'n_seeds': len(metrics_list),
        'test_accuracy': _summary(test_acc),
        'test_accuracy_ci_95': ci,
        'test_loss': _summary(test_loss),
        'training_time_seconds': _summary(train_time),
        'mean_param_grad_variance': _summary(param_grad_var),
        'n_parameters': int(metrics_list[0]['n_parameters']),
    }
    return stats


def find_seed_dirs(base_dir: Path) -> list:
    """Find all seed result directories under a base directory."""
    base_dir = Path(base_dir)
    seed_dirs = []

    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith('seed_'):
            if (item / 'metrics.json').exists():
                seed_dirs.append(item)

    if not seed_dirs and (base_dir / 'metrics.json').exists():
        seed_dirs.append(base_dir)

    return sorted(seed_dirs)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze variance across seed triples'
    )
    parser.add_argument(
        'path', type=str,
        help='Path to results directory containing seed_<N> subdirectories',
    )
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()

    result_path = Path(args.path)
    if not result_path.exists():
        print(f"ERROR: Path not found: {args.path}")
        sys.exit(1)

    seed_dirs = find_seed_dirs(result_path)

    if not seed_dirs:
        print(f"No seed results found in: {args.path}")
        sys.exit(1)

    metrics_list = []
    for seed_dir in seed_dirs:
        metrics = load_metrics(seed_dir)
        if metrics is None:
            print(f"WARNING: no metrics.json in {seed_dir}; skipped")
            continue
        missing = [k for k in REQUIRED_FIELDS if k not in metrics]
        if missing:
            print(
                f"ERROR: {seed_dir / 'metrics.json'} missing required fields: "
                f"{missing}"
            )
            sys.exit(1)
        metrics_list.append(metrics)

    if not metrics_list:
        print(f"No valid metrics.json files found in: {args.path}")
        sys.exit(1)

    stats = analyze_variance(metrics_list)

    if args.json:
        print(json.dumps(stats, indent=2))
        sys.exit(0)

    print(f"\n{'='*60}")
    print(f"SEED VARIANCE ANALYSIS")
    print(f"{'='*60}")
    print(f"Path: {args.path}")
    print(f"Seeds analyzed: {stats['n_seeds']}")

    acc = stats['test_accuracy']
    print(f"\nTest Accuracy:")
    print(f"  Mean:  {acc['mean']*100:.2f}%")
    print(f"  Std:   {acc['std']*100:.2f}%")
    print(f"  Range: {acc['min']*100:.2f}% - {acc['max']*100:.2f}%")
    print(f"  95% CI: [{stats['test_accuracy_ci_95'][0]*100:.2f}%, "
          f"{stats['test_accuracy_ci_95'][1]*100:.2f}%]")

    loss = stats['test_loss']
    print(f"\nTest Loss:")
    print(f"  Mean:  {loss['mean']:.4f}")
    print(f"  Std:   {loss['std']:.4f}")

    tt = stats['training_time_seconds']
    print(f"\nTraining Time:")
    print(f"  Mean:  {tt['mean']/60:.1f} min")
    print(f"  Std:   {tt['std']/60:.1f} min")
    print(f"  Total: {tt['mean']*stats['n_seeds']/60:.1f} min")

    pgv = stats['mean_param_grad_variance']
    print(f"\nMean Parameter-Gradient Variance:")
    print(f"  Mean:  {pgv['mean']:.3e}")
    print(f"  Std:   {pgv['std']:.3e}")

    print(f"\nParameters: {stats['n_parameters']}")
    print(f"{'='*60}")

    sys.exit(0)


if __name__ == "__main__":
    main()