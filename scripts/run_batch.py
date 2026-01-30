#!/usr/bin/env python
"""
Batch experiment runner for Hybrid-QNNs project.
Runs multiple seeds for a given approach and configuration.

Usage:
    python scripts/run_batch.py baseline configs/baseline_4layer.yaml
    python scripts/run_batch.py layerwise configs/layerwise_4layer.yaml
    python scripts/run_batch.py local_cost configs/local_cost_4layer.yaml
"""

import subprocess
import sys
import argparse
from pathlib import Path
import time

# Default seeds for experiments
SEEDS = [42, 123, 456, 789, 101112]


def run_experiment(approach: str, config_path: str, seed: int) -> bool:
    """
    Run a single experiment with given approach, config, and seed.
    
    Returns:
        True if successful, False otherwise
    """
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
    cmd = [sys.executable, script, config_path, '--seed', str(seed)]
    
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*70}")
    
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
    parser = argparse.ArgumentParser(description='Run batch experiments with multiple seeds')
    parser.add_argument('approach', type=str, choices=['baseline', 'layerwise', 'local_cost'],
                        help='Experiment approach')
    parser.add_argument('config', type=str, help='Path to config YAML file')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS,
                        help=f'Random seeds to use (default: {SEEDS})')
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Continue running even if one seed fails')
    args = parser.parse_args()
    
    # Validate config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"BATCH EXPERIMENT RUNNER")
    print(f"{'='*70}")
    print(f"Approach: {args.approach}")
    print(f"Config: {args.config}")
    print(f"Seeds: {args.seeds}")
    print(f"Total experiments: {len(args.seeds)}")
    print(f"{'='*70}")
    
    start_time = time.time()
    successful = 0
    failed = 0
    
    for i, seed in enumerate(args.seeds, 1):
        print(f"\n[{i}/{len(args.seeds)}] Running seed {seed}")
        
        success = run_experiment(args.approach, args.config, seed)
        
        if success:
            successful += 1
            print(f"✓ Seed {seed} completed successfully")
        else:
            failed += 1
            print(f"✗ Seed {seed} FAILED")
            
            if not args.continue_on_error:
                print("\nAborting batch run due to failure.")
                print("Use --continue-on-error to continue despite failures.")
                break
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"BATCH RUN COMPLETE")
    print(f"{'='*70}")
    print(f"Successful: {successful}/{len(args.seeds)}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"{'='*70}")
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
