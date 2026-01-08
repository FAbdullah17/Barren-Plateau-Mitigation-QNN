#!/usr/bin/env python
"""
Validate experiment results for completeness and correctness.

Usage:
    python scripts/validate_results.py results/
    python scripts/validate_results.py results/baseline/depth_4/seed_42/
"""

import json
import argparse
from pathlib import Path
import sys


REQUIRED_FIELDS = [
    'test_acc',
    'train_acc',
    'training_time',
    'gradient_stats',
    'barren_plateau_detected',
    'config',
    'seed'
]

REQUIRED_FILES = [
    'metrics.json',
    'training_history.png'
]


def validate_experiment(result_dir: Path) -> tuple:
    """
    Check if a single experiment completed successfully.
    
    Returns:
        (is_valid, message)
    """
    result_dir = Path(result_dir)
    
    # Check required files exist
    for filename in REQUIRED_FILES:
        filepath = result_dir / filename
        if not filepath.exists():
            return False, f"Missing file: {filename}"
    
    # Validate metrics.json contents
    metrics_path = result_dir / "metrics.json"
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in metrics.json: {e}"
    
    # Check required fields
    # Handle both naming conventions (test_acc vs test_accuracy)
    field_aliases = {
        'test_acc': ['test_acc', 'final_val_acc'],
        'train_acc': ['train_acc', 'final_train_acc'],
    }
    
    for field in REQUIRED_FIELDS:
        aliases = field_aliases.get(field, [field])
        if not any(alias in metrics for alias in aliases):
            return False, f"Missing field: {field}"
    
    # Check for reasonable values
    test_acc = metrics.get('test_acc', metrics.get('final_val_acc', -1))
    if not (0 <= test_acc <= 1):
        return False, f"Invalid test accuracy: {test_acc}"
    
    training_time = metrics.get('training_time', -1)
    if training_time < 0:
        return False, f"Invalid training time: {training_time}"
    
    return True, "Valid"


def find_experiment_dirs(base_dir: Path) -> list:
    """Find all experiment result directories."""
    base_dir = Path(base_dir)
    experiments = []
    
    # Check if this is already an experiment directory
    if (base_dir / 'metrics.json').exists():
        return [base_dir]
    
    # Search recursively for metrics.json files
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
    
    # Find all experiments
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
    
    # Summary
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
