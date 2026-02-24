#!/usr/bin/env python
"""
Run all 8-layer experiments (15 total: 3 approaches × 5 seeds).

Executes baseline, layerwise, and local cost training at 8-layer depth
across all random seeds. The 8-layer depth is critical for demonstrating
barren plateau severity — baseline training is expected to stagnate while
mitigation strategies (layerwise, local cost) should maintain trainability.

Usage:
    python scripts/run_8layer_experiments.py
    python scripts/run_8layer_experiments.py --seeds 42 123  # Run specific seeds only
"""

import subprocess
import sys
import argparse
import time
from pathlib import Path

DEFAULT_SEEDS = [42, 123, 456, 789, 101112]

EXPERIMENTS = [
    ('baseline', 'configs/baseline_8layer.yaml'),
    ('layerwise', 'configs/layerwise_8layer.yaml'),
    ('local_cost', 'configs/local_cost_8layer.yaml'),
]


def run_experiment(approach: str, config: str, seed: int) -> bool:
    """Run a single experiment."""
    script_map = {
        'baseline': 'experiments/run_baseline.py',
        'layerwise': 'experiments/run_layerwise.py',
        'local_cost': 'experiments/run_local_cost.py'
    }
    
    cmd = [sys.executable, script_map[approach], config, '--seed', str(seed)]
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False


def main():
    parser = argparse.ArgumentParser(description='Run all 8-layer experiments')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                        help=f'Seeds to run (default: {DEFAULT_SEEDS})')
    parser.add_argument('--approach', type=str, choices=['baseline', 'layerwise', 'local_cost'],
                        help='Run only specific approach')
    parser.add_argument('--dry-run', action='store_true', help='Show what would run without executing')
    args = parser.parse_args()
    
    experiments = EXPERIMENTS
    if args.approach:
        experiments = [(a, c) for a, c in EXPERIMENTS if a == args.approach]
    
    total = len(experiments) * len(args.seeds)
    
    print("="*70)
    print("8-LAYER EXPERIMENTS")
    print("="*70)
    print(f"Approaches: {len(experiments)}")
    print(f"Seeds: {args.seeds}")
    print(f"Total experiments: {total}")
    print(f"Estimated time: {total * 30} - {total * 60} minutes")
    print("")
    print("NOTE: 8-layer baseline expected to show barren plateau symptoms!")
    print("      Layerwise should maintain good performance.")
    print("="*70)
    
    if args.dry_run:
        print("\n[DRY RUN] Would execute:")
        for approach, config in experiments:
            for seed in args.seeds:
                print(f"  python experiments/run_{approach}.py {config} --seed {seed}")
        return
    
    start_time = time.time()
    completed = 0
    failed = 0
    
    for approach, config in experiments:
        print(f"\n{'='*70}")
        print(f"APPROACH: {approach.upper()}")
        print(f"{'='*70}")
        
        for seed in args.seeds:
            completed += 1
            print(f"\n[{completed}/{total}] Running {approach} with seed {seed}...")
            
            success = run_experiment(approach, config, seed)
            
            if success:
                print(f"✓ {approach} seed {seed} completed")
            else:
                failed += 1
                print(f"✗ {approach} seed {seed} FAILED")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("8-LAYER EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Completed: {completed - failed}/{total}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print("="*70)
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
