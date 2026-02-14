#!/usr/bin/env python
"""
System readiness check for Hybrid-QNNs experiments.

Verifies:
- Disk space availability
- Python environment and packages
- GPU status
- Time estimates for full experiment suite

Usage:
    python scripts/check_system.py
"""

import sys
import os
import shutil
from pathlib import Path

def check_disk_space(path: str, min_gb: float = 2.0) -> tuple:
    """Check available disk space."""
    try:
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        return free_gb >= min_gb, free_gb, total_gb
    except Exception as e:
        return False, 0, 0


def check_python_packages() -> dict:
    """Check required Python packages."""
    packages = {}
    
    try:
        import tensorflow as tf
        packages['tensorflow'] = tf.__version__
    except ImportError:
        packages['tensorflow'] = None
    
    try:
        import tensorflow_quantum as tfq
        packages['tensorflow_quantum'] = tfq.__version__
    except ImportError:
        packages['tensorflow_quantum'] = None
    
    try:
        import cirq
        packages['cirq'] = cirq.__version__
    except ImportError:
        packages['cirq'] = None
    
    try:
        import numpy as np
        packages['numpy'] = np.__version__
    except ImportError:
        packages['numpy'] = None
    
    try:
        import yaml
        packages['pyyaml'] = 'installed'
    except ImportError:
        packages['pyyaml'] = None
    
    return packages


def check_gpu() -> tuple:
    """Check GPU availability."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        return len(gpus), gpus
    except Exception:
        return 0, []


def check_results_directory() -> dict:
    """Check existing results."""
    results_path = Path('results')
    if not results_path.exists():
        return {'exists': False, 'experiments': 0}
    
    experiments = list(results_path.rglob('metrics.json'))
    return {'exists': True, 'experiments': len(experiments)}


def main():
    print("\n" + "="*60)
    print("SYSTEM READINESS CHECK")
    print("="*60)
    
    all_ok = True
    
    # 1. Disk Space
    print("\n📁 DISK SPACE")
    print("-" * 40)
    project_path = Path('.').resolve()
    ok, free_gb, total_gb = check_disk_space(str(project_path))
    
    if ok:
        print(f"✓ Available: {free_gb:.1f} GB / {total_gb:.1f} GB total")
    else:
        print(f"✗ Only {free_gb:.1f} GB available (need 2+ GB)")
        all_ok = False
    
    # 2. Python Environment
    print("\n🐍 PYTHON ENVIRONMENT")
    print("-" * 40)
    print(f"Python: {sys.version.split()[0]}")
    
    packages = check_python_packages()
    for pkg, version in packages.items():
        if version:
            print(f"✓ {pkg}: {version}")
        else:
            print(f"✗ {pkg}: NOT INSTALLED")
            all_ok = False
    
    # 3. GPU Status
    print("\n🎮 GPU STATUS")
    print("-" * 40)
    gpu_count, gpus = check_gpu()
    if gpu_count > 0:
        print(f"✓ GPUs available: {gpu_count}")
        for gpu in gpus:
            print(f"  - {gpu.name}")
    else:
        print("ℹ No GPU detected (CPU mode - this is OK for TFQ)")
    
    # 4. Results Directory
    print("\n📊 RESULTS DIRECTORY")
    print("-" * 40)
    results_info = check_results_directory()
    if results_info['exists']:
        print(f"✓ Results directory exists")
        print(f"  Existing experiments: {results_info['experiments']}")
    else:
        print("ℹ No results directory yet (will be created)")
    
    # 5. Time Estimates
    print("\n⏱️ TIME ESTIMATES")
    print("-" * 40)
    print("Based on 4-layer experiments (~20 min each):")
    print("")
    print("  4-layer (15 runs):  ~5-8 hours")
    print("  6-layer (15 runs):  ~8-12 hours")
    print("  8-layer (15 runs):  ~12-20 hours")
    print("  ─────────────────────────────────")
    print("  ALL 45 runs:        ~25-40 hours")
    print("")
    print("Recommendation: Run overnight or over weekend")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if all_ok:
        print("✓ System is READY for production experiments!")
    else:
        print("✗ System has issues that need to be resolved")
    
    print("="*60)
    
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
