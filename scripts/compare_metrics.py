#!/usr/bin/env python
"""
Compare two metrics.json files for reproducibility testing.

Compares test accuracy/loss, the gradient diagnostic (mean parameter-gradient
variance), parameter count, update budget, seed triple, and the resolved
config, using absolute/relative tolerances.

Usage:
    python scripts/compare_metrics.py results/baseline/depth_8/seed_0/metrics.json \
        results/baseline/depth_8/seed_0/metrics.json
"""

import json
import argparse
import sys
from pathlib import Path


def load_metrics(filepath: str) -> dict:
    """Load metrics from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def _relative_diff(a, b):
    """Relative difference |a-b|/|a| (0 when a == 0)."""
    if a == 0:
        return abs(b)
    return abs(a - b) / abs(a)


def compare_metrics(metrics1: dict, metrics2: dict, tolerance: dict = None) -> list:
    """Compare two metrics dicts; return the list of field comparisons."""
    if tolerance is None:
        tolerance = {
            'test_acc': 0.001,              # absolute
            'test_loss': 0.05,              # relative
            'mean_param_grad_variance': 0.05,  # relative
            'training_time_seconds': 0.1,   # relative
        }

    differences = []

    # Structural identity (must match exactly).
    for field, key in [('approach', 'config'), ('n_qubits', 'config'),
                       ('n_layers', 'config')]:
        v1 = metrics1['config'].get(field)
        v2 = metrics2['config'].get(field)
        differences.append({
            'field': f'config.{field}',
            'value1': v1,
            'value2': v2,
            'status': 'MATCH' if v1 == v2 else 'MISMATCH',
        })

    for field in ('seed_index', 'total_updates', 'n_parameters'):
        v1 = metrics1.get(field)
        v2 = metrics2.get(field)
        differences.append({
            'field': field,
            'value1': v1,
            'value2': v2,
            'status': 'MATCH' if v1 == v2 else 'MISMATCH',
        })

    # Test accuracy (absolute tolerance).
    acc1 = metrics1['test_acc']
    acc2 = metrics2['test_acc']
    differences.append({
        'field': 'test_acc',
        'value1': acc1,
        'value2': acc2,
        'difference': abs(acc1 - acc2),
        'threshold': tolerance['test_acc'],
        'status': 'MATCH' if abs(acc1 - acc2) <= tolerance['test_acc'] else 'MISMATCH',
    })

    # Test loss (relative tolerance).
    loss1 = metrics1['test_loss']
    loss2 = metrics2['test_loss']
    loss_rel = _relative_diff(loss1, loss2)
    differences.append({
        'field': 'test_loss',
        'value1': f"{loss1:.6f}",
        'value2': f"{loss2:.6f}",
        'difference': f"{loss_rel*100:.2f}%",
        'threshold': f"{tolerance['test_loss']*100:.0f}%",
        'status': 'MATCH' if loss_rel <= tolerance['test_loss'] else 'MISMATCH',
    })

    # Training time (relative tolerance).
    time1 = metrics1['training_time_seconds']
    time2 = metrics2['training_time_seconds']
    time_rel = _relative_diff(time1, time2)
    differences.append({
        'field': 'training_time_seconds',
        'value1': f"{time1:.2f}s",
        'value2': f"{time2:.2f}s",
        'difference': f"{time_rel*100:.1f}%",
        'threshold': f"{tolerance['training_time_seconds']*100:.0f}%",
        'status': 'MATCH' if time_rel <= tolerance['training_time_seconds'] else 'MISMATCH',
    })

    # Gradient diagnostic (relative tolerance).
    pgv1 = metrics1['training_diagnostic']['mean_param_grad_variance']
    pgv2 = metrics2['training_diagnostic']['mean_param_grad_variance']
    pgv_rel = _relative_diff(pgv1, pgv2)
    differences.append({
        'field': 'mean_param_grad_variance',
        'value1': f"{pgv1:.6e}",
        'value2': f"{pgv2:.6e}",
        'difference': f"{pgv_rel*100:.2f}%",
        'threshold': f"{tolerance['mean_param_grad_variance']*100:.0f}%",
        'status': 'MATCH' if pgv_rel <= tolerance['mean_param_grad_variance'] else 'MISMATCH',
    })

    return differences


def main():
    parser = argparse.ArgumentParser(
        description='Compare two experiment metrics for reproducibility'
    )
    parser.add_argument('file1', type=str, help='First metrics.json file')
    parser.add_argument('file2', type=str, help='Second metrics.json file')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()

    for filepath in [args.file1, args.file2]:
        if not Path(filepath).exists():
            print(f"ERROR: File not found: {filepath}")
            sys.exit(1)

    metrics1 = load_metrics(args.file1)
    metrics2 = load_metrics(args.file2)

    for name, metrics in (('file1', metrics1), ('file2', metrics2)):
        missing = [k for k in ('test_acc', 'test_loss', 'training_time_seconds',
                               'total_updates', 'n_parameters',
                               'training_diagnostic', 'config')
                   if k not in metrics]
        if missing:
            print(f"ERROR: {name} is missing required fields: {missing}")
            sys.exit(1)

    differences = compare_metrics(metrics1, metrics2)

    if args.json:
        print(json.dumps(differences, indent=2))
        sys.exit(0)

    print(f"\n{'='*70}")
    print(f"METRICS COMPARISON")
    print(f"{'='*70}")
    print(f"File 1: {args.file1}")
    print(f"File 2: {args.file2}")
    print(f"{'='*70}\n")

    print(f"{'Field':<28} {'Value 1':<20} {'Value 2':<20} {'Status':<10}")
    print("-" * 78)

    all_match = True
    for diff in differences:
        status = diff['status']
        if status == 'MISMATCH':
            all_match = False
            status_str = '✗ MISMATCH'
        else:
            status_str = '✓ MATCH'
        print(f"{diff['field']:<28} {str(diff['value1']):<20} "
              f"{str(diff['value2']):<20} {status_str:<10}")

    print("-" * 78)

    if all_match:
        print("\n✓ Results are REPRODUCIBLE (within tolerance)")
        sys.exit(0)
    else:
        print("\n✗ Results have DIFFERENCES beyond tolerance")
        sys.exit(1)


if __name__ == "__main__":
    main()