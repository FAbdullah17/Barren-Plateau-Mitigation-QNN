#!/usr/bin/env python
"""
Data Pipeline Consistency Tests for Barren-Plateau-Mitigation-QNN.

Verifies that data loading and quantum encoding are deterministic
and consistent across multiple calls with the same seed.

Run with: python -m pytest tests/test_data_consistency.py -v
"""

import numpy as np
import tensorflow as tf
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_mnist_loading_determinism():
    """Test that MNIST loading is deterministic with same seed."""
    from src.data import load_mnist_binary
    
    # Load data twice with same seed
    X1, y1, X1_test, y1_test = load_mnist_binary(
        digit1=3, digit2=6, train_size=100, test_size=50, seed=42
    )
    X2, y2, X2_test, y2_test = load_mnist_binary(
        digit1=3, digit2=6, train_size=100, test_size=50, seed=42
    )
    
    # Verify identical
    assert np.array_equal(X1, X2), "Training data not deterministic"
    assert np.array_equal(y1, y2), "Training labels not deterministic"
    assert np.array_equal(X1_test, X2_test), "Test data not deterministic"
    assert np.array_equal(y1_test, y2_test), "Test labels not deterministic"
    
    print("✓ MNIST loading is deterministic")


def test_mnist_different_seeds():
    """Test that different seeds produce different data."""
    from src.data import load_mnist_binary
    
    X1, y1, _, _ = load_mnist_binary(
        digit1=3, digit2=6, train_size=100, test_size=50, seed=42
    )
    X2, y2, _, _ = load_mnist_binary(
        digit1=3, digit2=6, train_size=100, test_size=50, seed=123
    )
    
    # Data should be different with different seeds
    assert not np.array_equal(X1, X2), "Different seeds should give different data"
    
    print("✓ Different seeds produce different data")


def test_data_normalization():
    """Test that data is properly normalized to [0, 1] range."""
    from src.data import load_mnist_binary
    
    X_train, _, X_test, _ = load_mnist_binary(
        digit1=3, digit2=6, train_size=100, test_size=50, seed=42
    )
    
    # Check normalization
    assert X_train.min() >= 0.0, f"Training min {X_train.min()} < 0"
    assert X_train.max() <= 1.0, f"Training max {X_train.max()} > 1"
    assert X_test.min() >= 0.0, f"Test min {X_test.min()} < 0"
    assert X_test.max() <= 1.0, f"Test max {X_test.max()} > 1"
    
    print("✓ Data correctly normalized to [0, 1]")


def test_label_binary():
    """Test that labels are binary (0 or 1)."""
    from src.data import load_mnist_binary
    
    _, y_train, _, y_test = load_mnist_binary(
        digit1=3, digit2=6, train_size=100, test_size=50, seed=42
    )
    
    # Check binary labels
    assert set(np.unique(y_train)).issubset({0, 1}), f"Non-binary train labels: {np.unique(y_train)}"
    assert set(np.unique(y_test)).issubset({0, 1}), f"Non-binary test labels: {np.unique(y_test)}"
    
    print("✓ Labels are binary (0, 1)")


def test_data_shape():
    """Test that data has correct shape for quantum encoding."""
    from src.data import load_mnist_binary
    
    X_train, y_train, X_test, y_test = load_mnist_binary(
        digit1=3, digit2=6, train_size=100, test_size=50, 
        image_size=(4, 4), seed=42
    )
    
    # Check shapes
    assert X_train.shape == (100, 16), f"Train shape {X_train.shape} != (100, 16)"
    assert X_test.shape == (50, 16), f"Test shape {X_test.shape} != (50, 16)"
    assert y_train.shape == (100,), f"Train labels shape {y_train.shape} != (100,)"
    assert y_test.shape == (50,), f"Test labels shape {y_test.shape} != (50,)"
    
    print("✓ Data shapes correct for 4x4 quantum encoding")


def test_quantum_circuit_encoding_determinism():
    """Test that quantum circuit encoding is deterministic."""
    import cirq
    import tensorflow_quantum as tfq
    from src.data import load_mnist_binary
    
    def encode_to_circuits(data, n_qubits=4):
        """Encode data to quantum circuits."""
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuits = []
        for sample in data:
            circuit = cirq.Circuit()
            angles = sample * np.pi
            for i, qubit in enumerate(qubits):
                if i < len(angles):
                    circuit.append(cirq.ry(angles[i])(qubit))
            circuits.append(circuit)
        return circuits
    
    X, _, _, _ = load_mnist_binary(
        digit1=3, digit2=6, train_size=10, test_size=5, seed=42
    )
    
    # Encode twice
    circuits1 = encode_to_circuits(X)
    circuits2 = encode_to_circuits(X)
    
    # Compare circuit representations
    for c1, c2 in zip(circuits1, circuits2):
        assert str(c1) == str(c2), "Quantum encoding not deterministic"
    
    print("✓ Quantum circuit encoding is deterministic")


def test_train_test_split_balance():
    """Test that train/test split maintains class balance."""
    from src.data import load_mnist_binary
    
    _, y_train, _, y_test = load_mnist_binary(
        digit1=3, digit2=6, train_size=100, test_size=50, seed=42
    )
    
    # Check class balance (should be roughly 50/50)
    train_balance = np.mean(y_train)
    test_balance = np.mean(y_test)
    
    # Allow some variance (0.3 to 0.7)
    assert 0.3 <= train_balance <= 0.7, f"Train imbalanced: {train_balance:.2f}"
    assert 0.3 <= test_balance <= 0.7, f"Test imbalanced: {test_balance:.2f}"
    
    print(f"✓ Class balance - Train: {train_balance:.2f}, Test: {test_balance:.2f}")


if __name__ == "__main__":
    print("="*60)
    print("DATA PIPELINE CONSISTENCY TESTS")
    print("="*60)
    
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    tests = [
        test_mnist_loading_determinism,
        test_mnist_different_seeds,
        test_data_normalization,
        test_label_binary,
        test_data_shape,
        test_quantum_circuit_encoding_determinism,
        test_train_test_split_balance,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\nRunning {test.__name__}...")
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    sys.exit(0 if failed == 0 else 1)
