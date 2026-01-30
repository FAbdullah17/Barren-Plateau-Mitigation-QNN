#!/usr/bin/env python
"""
Compare two metrics.json files for reproducibility testing.

Usage:
    python scripts/compare_metrics.py results/run1/metrics.json results/run2/metrics.json
"""

import json
import argparse
import sys
from pathlib import Path


def load_metrics(filepath: str) -> dict:
    """Load metrics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_metrics(metrics1: dict, metrics2: dict, tolerance: dict = None) -> list:
    """
    Compare two metrics dictionaries.
    
    Returns:
        List of differences found
    """
    if tolerance is None:
        tolerance = {
            'test_acc': 0.001,      # 0.1% tolerance
            'train_acc': 0.001,
            'training_time': 0.1,   # 10% tolerance (relative)
            'gradient_norm': 0.01   # 1% tolerance (relative)
        }
    
    differences = []
    
    # Compare key metrics
    acc_key1 = 'test_acc' if 'test_acc' in metrics1 else 'final_val_acc'
    acc_key2 = 'test_acc' if 'test_acc' in metrics2 else 'final_val_acc'
    
    acc1 = metrics1.get(acc_key1, 0)
    acc2 = metrics2.get(acc_key2, 0)
    acc_diff = abs(acc1 - acc2)
    
    if acc_diff > tolerance['test_acc']:
        differences.append({
            'field': 'test_accuracy',
            'value1': acc1,
            'value2': acc2,
            'difference': acc_diff,
            'threshold': tolerance['test_acc'],
            'status': 'MISMATCH'
        })
    else:
        differences.append({
            'field': 'test_accuracy',
            'value1': acc1,
            'value2': acc2,
            'difference': acc_diff,
            'threshold': tolerance['test_acc'],
            'status': 'MATCH'
        })
    
    # Compare training time (relative difference)
    time1 = metrics1.get('training_time', 0)
    time2 = metrics2.get('training_time', 0)
    if time1 > 0:
        time_diff_rel = abs(time1 - time2) / time1
    else:
        time_diff_rel = 0
    
    differences.append({
        'field': 'training_time',
        'value1': f"{time1:.2f}s",
        'value2': f"{time2:.2f}s",
        'difference': f"{time_diff_rel*100:.1f}%",
        'threshold': f"{tolerance['training_time']*100:.0f}%",
        'status': 'MATCH' if time_diff_rel <= tolerance['training_time'] else 'MISMATCH'
    })
    
    # Compare barren plateau detection
    bp1 = metrics1.get('barren_plateau_detected', False)
    bp2 = metrics2.get('barren_plateau_detected', False)
    differences.append({
        'field': 'barren_plateau_detected',
        'value1': bp1,
        'value2': bp2,
        'status': 'MATCH' if bp1 == bp2 else 'MISMATCH'
    })
    
    # Compare gradient statistics
    gs1 = metrics1.get('gradient_stats', {})
    gs2 = metrics2.get('gradient_stats', {})
    
    mean_norm1 = gs1.get('mean_norm', 0)
    mean_norm2 = gs2.get('mean_norm', 0)
    
    if mean_norm1 > 0:
        norm_diff_rel = abs(mean_norm1 - mean_norm2) / mean_norm1
    else:
        norm_diff_rel = 0
    
    differences.append({
        'field': 'mean_gradient_norm',
        'value1': f"{mean_norm1:.6e}",
        'value2': f"{mean_norm2:.6e}",
        'difference': f"{norm_diff_rel*100:.1f}%",
        'threshold': f"{tolerance['gradient_norm']*100:.0f}%",
        'status': 'MATCH' if norm_diff_rel <= tolerance['gradient_norm'] else 'MISMATCH'
    })
    
    return differences


def main():
    parser = argparse.ArgumentParser(description='Compare two experiment metrics for reproducibility')
    parser.add_argument('file1', type=str, help='First metrics.json file')
    parser.add_argument('file2', type=str, help='Second metrics.json file')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()
    
    # Validate files exist
    for filepath in [args.file1, args.file2]:
        if not Path(filepath).exists():
            print(f"ERROR: File not found: {filepath}")
            sys.exit(1)
    
    # Load metrics
    metrics1 = load_metrics(args.file1)
    metrics2 = load_metrics(args.file2)
    
    # Compare
    differences = compare_metrics(metrics1, metrics2)
    
    if args.json:
        print(json.dumps(differences, indent=2))
        sys.exit(0)
    
    # Print comparison table
    print(f"\n{'='*70}")
    print(f"METRICS COMPARISON")
    print(f"{'='*70}")
    print(f"File 1: {args.file1}")
    print(f"File 2: {args.file2}")
    print(f"{'='*70}\n")
    
    print(f"{'Field':<25} {'Value 1':<15} {'Value 2':<15} {'Status':<10}")
    print("-" * 70)
    
    all_match = True
    for diff in differences:
        status = diff['status']
        if status == 'MISMATCH':
            all_match = False
            status_str = '✗ MISMATCH'
        else:
            status_str = '✓ MATCH'
        
        print(f"{diff['field']:<25} {str(diff['value1']):<15} {str(diff['value2']):<15} {status_str:<10}")
    
    print("-" * 70)
    
    if all_match:
        print("\n✓ Results are REPRODUCIBLE (within tolerance)")
        sys.exit(0)
    else:
        print("\n✗ Results have DIFFERENCES beyond tolerance")
        sys.exit(1)


if __name__ == "__main__":
    main()
