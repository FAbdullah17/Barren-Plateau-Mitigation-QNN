#!/usr/bin/env python
"""
Analyze variance across different random seeds.

Usage:
    python scripts/analyze_seed_variance.py results/baseline/depth_4/
    python scripts/analyze_seed_variance.py results/ --approach baseline --depth 4
"""

import json
import argparse
from pathlib import Path
import sys

def load_metrics(result_dir: Path) -> dict:
    """Load metrics from a result directory."""
    metrics_path = result_dir / 'metrics.json'
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        return json.load(f)


def analyze_variance(result_dirs: list) -> dict:
    """Analyze variance across seed results."""
    metrics_list = []
    
    for dir_path in result_dirs:
        metrics = load_metrics(Path(dir_path))
        if metrics:
            metrics_list.append(metrics)
    
    if not metrics_list:
        return None
    
    # Extract key metrics
    accuracies = []
    train_accs = []
    grad_norms = []
    training_times = []
    
    for m in metrics_list:
        acc = m.get('test_acc', m.get('final_val_acc', 0))
        accuracies.append(acc)
        
        train_acc = m.get('train_acc', m.get('final_train_acc', 0))
        train_accs.append(train_acc)
        
        gs = m.get('gradient_stats', {})
        grad_norms.append(gs.get('mean_norm', 0))
        
        training_times.append(m.get('training_time', 0))
    
    # Calculate statistics
    import numpy as np
    
    return {
        'n_seeds': len(metrics_list),
        'test_accuracy': {
            'mean': float(np.mean(accuracies)),
            'std': float(np.std(accuracies)),
            'min': float(np.min(accuracies)),
            'max': float(np.max(accuracies)),
            'values': [float(x) for x in accuracies]
        },
        'train_accuracy': {
            'mean': float(np.mean(train_accs)),
            'std': float(np.std(train_accs)),
        },
        'gradient_norm': {
            'mean': float(np.mean(grad_norms)),
            'std': float(np.std(grad_norms)),
        },
        'training_time': {
            'mean': float(np.mean(training_times)),
            'std': float(np.std(training_times)),
            'total': float(np.sum(training_times)),
        }
    }


def find_seed_dirs(base_dir: Path) -> list:
    """Find all seed directories under a base directory."""
    base_dir = Path(base_dir)
    seed_dirs = []
    
    # Check for seed_* subdirectories
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith('seed_'):
            if (item / 'metrics.json').exists():
                seed_dirs.append(item)
    
    # If no seed dirs found, check if base_dir itself has metrics
    if not seed_dirs and (base_dir / 'metrics.json').exists():
        seed_dirs.append(base_dir)
    
    return sorted(seed_dirs)


def main():
    parser = argparse.ArgumentParser(description='Analyze variance across random seeds')
    parser.add_argument('path', type=str, help='Path to results directory containing seed subdirs')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()
    
    result_path = Path(args.path)
    if not result_path.exists():
        print(f"ERROR: Path not found: {args.path}")
        sys.exit(1)
    
    # Find seed directories
    seed_dirs = find_seed_dirs(result_path)
    
    if not seed_dirs:
        print(f"No seed results found in: {args.path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"SEED VARIANCE ANALYSIS")
    print(f"{'='*60}")
    print(f"Path: {args.path}")
    print(f"Seeds found: {len(seed_dirs)}")
    for sd in seed_dirs:
        print(f"  - {sd.name}")
    
    # Analyze
    stats = analyze_variance(seed_dirs)
    
    if not stats:
        print("\nERROR: Could not analyze metrics")
        sys.exit(1)
    
    if args.json:
        print(json.dumps(stats, indent=2))
        sys.exit(0)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS ({stats['n_seeds']} seeds)")
    print(f"{'='*60}")
    
    acc = stats['test_accuracy']
    print(f"\nTest Accuracy:")
    print(f"  Mean:  {acc['mean']*100:.2f}%")
    print(f"  Std:   {acc['std']*100:.2f}%")
    print(f"  Range: {acc['min']*100:.2f}% - {acc['max']*100:.2f}%")
    
    print(f"\nTrain Accuracy:")
    ta = stats['train_accuracy']
    print(f"  Mean:  {ta['mean']*100:.2f}%")
    print(f"  Std:   {ta['std']*100:.2f}%")
    
    print(f"\nGradient Norm:")
    gn = stats['gradient_norm']
    print(f"  Mean:  {gn['mean']:.6e}")
    print(f"  Std:   {gn['std']:.6e}")
    
    print(f"\nTraining Time:")
    tt = stats['training_time']
    print(f"  Mean:  {tt['mean']/60:.1f} min")
    print(f"  Std:   {tt['std']/60:.1f} min")
    print(f"  Total: {tt['total']/60:.1f} min")
    
    # Variance check
    print(f"\n{'='*60}")
    print(f"VARIANCE CHECK")
    print(f"{'='*60}")
    
    variance_ok = acc['std'] < 0.05  # Less than 5%
    if variance_ok:
        print(f"✓ Accuracy variance {acc['std']*100:.2f}% < 5% threshold")
    else:
        print(f"⚠ Accuracy variance {acc['std']*100:.2f}% > 5% threshold")
    
    print(f"{'='*60}")
    
    sys.exit(0 if variance_ok else 1)


if __name__ == "__main__":
    main()
