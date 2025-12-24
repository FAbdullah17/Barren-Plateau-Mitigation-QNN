"""Setup results directory structure for experiments.

Developer Assignment (Weeks 1-2):
    Primary: Frahan Riaz - Results directory structure
"""

from pathlib import Path
from src.utils.constants import APPROACHES, MULTI_DEPTH_EXPERIMENTS, DEFAULT_SEEDS, RESULTS_DIR


def setup_results_directories(
    base_dir: str = RESULTS_DIR,
    create_seed_dirs: bool = False
) -> list:
    """
    Create the complete results directory structure.
    
    Args:
        base_dir: Base results directory (default: "results")
        create_seed_dirs: If True, also create seed subdirectories
    
    Returns:
        List of created directory paths
    """
    base_path = Path(base_dir)
    
    created_dirs = []
    
    # Create directories for each approach and depth
    for approach in APPROACHES:
        for depth in MULTI_DEPTH_EXPERIMENTS:
            # Main depth directory
            depth_dir = base_path / approach / f"depth_{depth}"
            depth_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.append(depth_dir)
            
            # Optionally create seed subdirectories
            if create_seed_dirs:
                for seed in DEFAULT_SEEDS:
                    seed_dir = depth_dir / f"seed_{seed}"
                    seed_dir.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(seed_dir)
    
    # Create comparison directory for final analysis
    comparison_dir = base_path / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    created_dirs.append(comparison_dir)
    
    return created_dirs


def print_structure(base_dir: str = RESULTS_DIR) -> None:
    """Print the created directory structure."""
    base_path = Path(base_dir)
    
    print("\nResults Directory Structure:")
    print("=" * 60)
    
    for approach in APPROACHES:
        print(f"\n{approach}/")
        for depth in MULTI_DEPTH_EXPERIMENTS:
            depth_path = base_path / approach / f"depth_{depth}"
            status = "✓" if depth_path.exists() else "✗"
            print(f"  {status} depth_{depth}/")
            if depth_path.exists():
                seed_dirs = list(depth_path.glob("seed_*"))
                if seed_dirs:
                    for seed_dir in sorted(seed_dirs):
                        print(f"      {seed_dir.name}/")
    
    comparison_path = base_path / "comparison"
    status = "✓" if comparison_path.exists() else "✗"
    print(f"\n{status} comparison/")


if __name__ == "__main__":
    print("Setting up results directory structure...")
    
    # Create directories with seed subdirectories
    created = setup_results_directories(create_seed_dirs=True)
    
    print(f"\n✓ Created {len(created)} directories")
    print(f"  Base directory: {Path(RESULTS_DIR).absolute()}")
    
    # Print structure
    print_structure()
    
    print("\n✓ Results directory structure ready!")