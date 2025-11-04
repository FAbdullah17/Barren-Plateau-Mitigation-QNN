"""Quick test to verify installation and setup."""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import tensorflow_quantum as tfq
        print(f"✓ TensorFlow Quantum {tfq.__version__}")
    except ImportError as e:
        print(f"✗ TensorFlow Quantum import failed: {e}")
        return False
    
    try:
        import cirq
        print(f"✓ Cirq {cirq.__version__}")
    except ImportError as e:
        print(f"✗ Cirq import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import yaml
        print(f"✓ PyYAML")
    except ImportError as e:
        print(f"✗ PyYAML import failed: {e}")
        return False
    
    return True


def test_project_structure():
    """Test that project structure is correct."""
    print("\nTesting project structure...")
    
    from pathlib import Path
    
    required_dirs = [
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "experiments",
        "configs",
        "results",
        "notebooks",
        "tests",
        "docs"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ not found")
            all_exist = False
    
    return all_exist


def test_src_imports():
    """Test that source modules can be imported."""
    print("\nTesting source modules...")
    
    try:
        from src.data import load_mnist_binary
        print("✓ src.data")
    except ImportError as e:
        print(f"✗ src.data import failed: {e}")
        return False
    
    try:
        from src.models import QuantumCircuit, QuantumNeuralNetwork
        print("✓ src.models")
    except ImportError as e:
        print(f"✗ src.models import failed: {e}")
        return False
    
    try:
        from src.training import BaselineTrainer, LayerwiseTrainer
        print("✓ src.training")
    except ImportError as e:
        print(f"✗ src.training import failed: {e}")
        return False
    
    try:
        from src.evaluation import GradientTracker, plot_training_history
        print("✓ src.evaluation")
    except ImportError as e:
        print(f"✗ src.evaluation import failed: {e}")
        return False
    
    return True


def test_simple_circuit():
    """Test basic quantum circuit creation."""
    print("\nTesting quantum circuit creation...")
    
    try:
        import cirq
        from src.models import QuantumCircuit
        
        qc = QuantumCircuit(n_qubits=4, n_layers=2)
        circuit = qc.get_circuit()
        params = qc.get_parameters()
        
        print(f"✓ Created circuit with {len(params)} parameters")
        print(f"✓ Circuit has {len(circuit)} moments")
        
        return True
    except Exception as e:
        print(f"✗ Circuit creation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("SETUP VERIFICATION TEST")
    print("="*70)
    print(f"Python version: {sys.version}")
    print("="*70)
    
    results = []
    
    # Test imports
    results.append(("Package imports", test_imports()))
    
    # Test project structure
    results.append(("Project structure", test_project_structure()))
    
    # Test source imports
    results.append(("Source modules", test_src_imports()))
    
    # Test circuit creation
    results.append(("Quantum circuit", test_simple_circuit()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("="*70)
    print(f"Results: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n✓ All tests passed! Your environment is ready.")
        print("\nNext steps:")
        print("  1. Review configs/ for experiment configurations")
        print("  2. Run: python experiments/run_baseline.py")
        print("  3. Run: python experiments/run_comparison.py")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("  1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("  2. Verify Python 3.10 is being used")
        print("  3. Check that you're in the project root directory")
        return 1


if __name__ == "__main__":
    sys.exit(main())
