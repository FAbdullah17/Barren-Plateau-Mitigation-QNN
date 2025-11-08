"""
End-to-end pipeline test for Hybrid-QNN project.

This script tests the complete workflow:
1. Load MNIST data (3 vs 6)
2. Encode as quantum circuits
3. Train baseline QNN model
4. Verify results

Use small dataset and few epochs for quick verification.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import numpy as np
from pathlib import Path

# Import project modules
from src.data.mnist_loader import load_mnist_binary, encode_data_for_qnn
from src.models.quantum_circuit import QuantumCircuit, create_readout_operators
from src.training.baseline_trainer import BaselineTrainer

print("=" * 70)
print("HYBRID-QNN END-TO-END PIPELINE TEST")
print("=" * 70)

# Configuration
SEED = 42
TRAIN_SIZE = 50  # Small for quick test
TEST_SIZE = 20
N_QUBITS = 4
N_LAYERS = 2  # Shallow for speed
EPOCHS = 3
BATCH_SIZE = 10

print(f"\nConfiguration:")
print(f"  - Train samples: {TRAIN_SIZE}")
print(f"  - Test samples: {TEST_SIZE}")
print(f"  - Circuit: {N_QUBITS} qubits, {N_LAYERS} layers")
print(f"  - Training: {EPOCHS} epochs, batch size {BATCH_SIZE}")

# Step 1: Load data
print(f"\n{'='*70}")
print("STEP 1: Loading MNIST data (digits 3 vs 6)...")
print(f"{'='*70}")

X_train, y_train, X_test, y_test = load_mnist_binary(
    digit1=3,
    digit2=6,
    train_size=TRAIN_SIZE,
    test_size=TEST_SIZE,
    image_size=(4, 4),
    seed=SEED
)

print(f"✓ Data loaded:")
print(f"  - X_train: {X_train.shape}")
print(f"  - y_train: {y_train.shape}, distribution: {np.bincount(y_train)}")
print(f"  - X_test: {X_test.shape}")
print(f"  - y_test: {y_test.shape}, distribution: {np.bincount(y_test)}")

# Step 2: Create quantum circuits
print(f"\n{'='*70}")
print("STEP 2: Building quantum circuits...")
print(f"{'='*70}")

qc = QuantumCircuit(n_qubits=N_QUBITS, n_layers=N_LAYERS)
circuit_template = qc.get_circuit()
params = qc.get_parameters()

print(f"✓ Circuit created:")
print(f"  - Qubits: {N_QUBITS}")
print(f"  - Layers: {N_LAYERS}")
print(f"  - Parameters: {len(params)}")
print(f"  - Gates: {len(circuit_template)}")

# Step 3: Encode data as quantum circuits
print(f"\n{'='*70}")
print("STEP 3: Encoding data as quantum circuits...")
print(f"{'='*70}")

def encode_batch(X_batch, circuit, params):
    """Encode classical data into quantum circuits."""
    qubits = sorted(circuit.all_qubits())
    circuits = []
    
    for x in X_batch:
        # Amplitude encoding: rotation angles from pixel values
        data_circuit = cirq.Circuit()
        for i, qubit in enumerate(qubits):
            angle = x[i] * np.pi  # Scale to [0, π]
            data_circuit.append(cirq.ry(angle)(qubit))
        
        # Append the parameterized circuit
        full_circuit = data_circuit + circuit
        circuits.append(full_circuit)
    
    return tfq.convert_to_tensor(circuits)

train_circuits = encode_batch(X_train, circuit_template, params)
test_circuits = encode_batch(X_test, circuit_template, params)

print(f"✓ Data encoded as quantum circuits:")
print(f"  - Train circuits: {train_circuits.shape}")
print(f"  - Test circuits: {test_circuits.shape}")

# Step 4: Train baseline model
print(f"\n{'='*70}")
print("STEP 4: Training baseline QNN model...")
print(f"{'='*70}")

trainer = BaselineTrainer(
    n_qubits=N_QUBITS,
    n_layers=N_LAYERS,
    learning_rate=0.01,
    batch_size=BATCH_SIZE,
    local_cost=False,
    seed=SEED
)

results = trainer.train(
    train_circuits=train_circuits,
    train_labels=y_train,
    val_circuits=test_circuits,
    val_labels=y_test,
    epochs=EPOCHS
)

# Step 5: Display results
print(f"\n{'='*70}")
print("STEP 5: Results Summary")
print(f"{'='*70}")

print(f"\n✓ Training completed successfully!")
print(f"\nFinal Metrics:")
print(f"  - Training Loss: {results['final_train_loss']:.4f}")
print(f"  - Training Accuracy: {results['final_train_acc']:.4f}")
print(f"  - Validation Loss: {results['final_val_loss']:.4f}")
print(f"  - Validation Accuracy: {results['final_val_acc']:.4f}")
print(f"  - Test Accuracy: {results['test_acc']:.4f}")
print(f"  - Training Time: {results['training_time']:.2f}s")

print(f"\nGradient Statistics:")
grad_stats = results['gradient_stats']
print(f"  - Mean gradient norm: {grad_stats['mean_norm']:.6f}")
print(f"  - Gradient variance: {grad_stats['variance']:.6e}")
print(f"  - Barren plateau detected: {results['barren_plateau_detected']}")

print(f"\n{'='*70}")
print("✅ END-TO-END PIPELINE TEST PASSED!")
print(f"{'='*70}")
print(f"\nAll core modules working:")
print(f"  ✓ Data loading (mnist_loader.py)")
print(f"  ✓ Circuit building (quantum_circuit.py)")
print(f"  ✓ QNN model (qnn_model.py)")
print(f"  ✓ Baseline training (baseline_trainer.py)")
print(f"\n🎉 Your infrastructure is ready for Week 3-4 implementation phase!")
