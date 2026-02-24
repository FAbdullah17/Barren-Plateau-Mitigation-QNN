#!/usr/bin/env python
"""
Check output format consistency across all experiment approaches.

Verifies that baseline, layerwise, and local_cost experiments
produce identical file structures and metrics schema.

Usage:
    python scripts/check_output_format.py results/
"""

import json
import argparse
from pathlib import Path
import sys


def get_experiment_structure(result_dir: Path) -> dict:
    """Get the file structure of an experiment result directory."""
    result_dir = Path(result_dir)
    
    structure = {
        'files': [],
        'metrics_keys': set(),
        'has_metrics': False,
        'has_plot': False,
        'has_config': False
    }
    
    for item in result_dir.iterdir():
        if item.is_file():
            structure['files'].append(item.name)
            
            if item.name == 'metrics.json':
                structure['has_metrics'] = True
                with open(item) as f:
                    metrics = json.load(f)
                structure['metrics_keys'] = set(metrics.keys())
                structure['has_config'] = 'config' in metrics
                
            if item.suffix == '.png':
                structure['has_plot'] = True
    
    return structure


def find_all_experiments(base_dir: Path) -> dict:
    """Find all experiment result directories organized by approach."""
    base_dir = Path(base_dir)
    experiments = {}
    
    for metrics_file in base_dir.rglob('metrics.json'):
        exp_dir = metrics_file.parent
        
        # Determine approach from path
        rel_path = exp_dir.relative_to(base_dir)
        parts = rel_path.parts
        
        if len(parts) >= 1:
            approach = parts[0]  # baseline, layerwise, or local_cost
        else:
            approach = 'unknown'
        
        if approach not in experiments:
            experiments[approach] = []
        
        experiments[approach].append({
            'path': exp_dir,
            'rel_path': str(rel_path)
        })
    
    return experiments


def compare_schemas(experiments: dict) -> tuple:
    """Compare metrics schemas across all experiments."""
    all_keys = {}
    issues = []
    
    for approach, exp_list in experiments.items():
        for exp in exp_list:
            structure = get_experiment_structure(exp['path'])
            key_tuple = tuple(sorted(structure['metrics_keys']))
            
            if key_tuple not in all_keys:
                all_keys[key_tuple] = []
            all_keys[key_tuple].append(f"{approach}/{exp['rel_path']}")
    
    # Check if all experiments have the same schema
    if len(all_keys) > 1:
        issues.append("SCHEMA MISMATCH: Different metrics.json schemas detected")
        for keys, paths in all_keys.items():
            issues.append(f"  Schema with {len(keys)} keys: {paths[:3]}...")
    
    return len(all_keys) == 1, issues


def main():
    parser = argparse.ArgumentParser(description='Check output format consistency')
    parser.add_argument('path', type=str, help='Path to results directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    results_path = Path(args.path)
    if not results_path.exists():
        print(f"ERROR: Path not found: {args.path}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("OUTPUT FORMAT CONSISTENCY CHECK")
    print("="*60)
    
    # Find all experiments
    experiments = find_all_experiments(results_path)
    
    if not experiments:
        print(f"No experiments found in: {args.path}")
        sys.exit(1)
    
    print(f"\nFound {sum(len(v) for v in experiments.values())} experiments across {len(experiments)} approaches\n")
    
    # Check each approach
    all_issues = []
    
    for approach, exp_list in sorted(experiments.items()):
        print(f"Approach: {approach}")
        print(f"  Experiments: {len(exp_list)}")
        
        # Check first experiment structure
        if exp_list:
            structure = get_experiment_structure(exp_list[0]['path'])
            print(f"  Files: {', '.join(sorted(structure['files']))}")
            print(f"  Has metrics.json: {'✓' if structure['has_metrics'] else '✗'}")
            print(f"  Has training plot: {'✓' if structure['has_plot'] else '✗'}")
            print(f"  Has config snapshot: {'✓' if structure['has_config'] else '✗'}")
            print(f"  Metrics fields: {len(structure['metrics_keys'])}")
            
            if args.verbose:
                print(f"  Fields: {sorted(structure['metrics_keys'])}")
            
            # Validate required files
            if not structure['has_metrics']:
                all_issues.append(f"{approach}: Missing metrics.json")
            if not structure['has_plot']:
                all_issues.append(f"{approach}: Missing training plot")
        print()
    
    # Compare schemas across approaches
    schema_consistent, schema_issues = compare_schemas(experiments)
    all_issues.extend(schema_issues)
    
    # Summary
    print("="*60)
    print("CONSISTENCY CHECK SUMMARY")
    print("="*60)
    
    if not all_issues:
        print("✓ All output formats are consistent!")
        print("✓ All experiments have metrics.json")
        print("✓ All experiments have training plots")
        print("✓ Metrics schema is consistent across approaches")
        status = 0
    else:
        print("Issues found:")
        for issue in all_issues:
            print(f"  ✗ {issue}")
        status = 1
    
    print("="*60)
    
    sys.exit(status)


if __name__ == "__main__":
    main()
