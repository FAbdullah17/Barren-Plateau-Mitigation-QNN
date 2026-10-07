"""End-to-end pipeline smoke test.

Loads MNIST 3-vs-6 through the fixed low-dimensional pipeline
(28x28 -> 4x4 -> 16 dims -> train-only PCA -> n_components), encodes exactly
``n_qubits`` features into circuits, and trains the baseline QNN for a few
gradient steps to verify the whole stack runs end to end. Kept small for speed.
"""

import pytest
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq

from src.data import load_mnist_binary, prepare_features, encode_data_for_qnn
from src.training.baseline_trainer import BaselineTrainer

SEED = 42
TRAIN_SIZE = 50
TEST_SIZE = 20
N_QUBITS = 4
N_LAYERS = 2
TOTAL_UPDATES = 4
BATCH_SIZE = 10
LOG_FREQUENCY = 2


def _convert_to_circuits(data, n_qubits):
    """Encode all n_qubits features; fail loudly on truncation."""
    assert data.shape[1] == n_qubits
    qubits = cirq.GridQubit.rect(1, n_qubits)
    circuits = []
    for sample in data:
        circuit = cirq.Circuit()
        angles = encode_data_for_qnn(sample)
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.ry(angles[i])(qubit))
        circuits.append(circuit)
    return tfq.convert_to_tensor(circuits)


def test_end_to_end_pipeline():
    X_train, y_train, X_test, y_test = load_mnist_binary(
        digit1=3,
        digit2=6,
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        data_seed=SEED,
    )
    X_train, X_test, pca_info = prepare_features(
        X_train, X_test, n_components=N_QUBITS
    )
    assert X_train.shape == (TRAIN_SIZE, N_QUBITS)
    assert X_test.shape == (TEST_SIZE, N_QUBITS)
    assert pca_info['n_components'] == N_QUBITS

    train_circuits = _convert_to_circuits(X_train, N_QUBITS)
    test_circuits = _convert_to_circuits(X_test, N_QUBITS)

    trainer = BaselineTrainer(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        cost='global',
        learning_rate=0.01,
        batch_size=BATCH_SIZE,
        total_updates=TOTAL_UPDATES,
        log_frequency=LOG_FREQUENCY,
        diagnostic_samples=10,
        init_seed=SEED,
        training_seed=SEED,
    )
    results = trainer.train(
        train_circuits=train_circuits,
        train_labels=y_train,
        val_circuits=test_circuits,
        val_labels=y_test,
    )

    for key in (
        'config', 'total_updates', 'n_parameters', 'test_loss', 'test_acc',
        'training_time_seconds', 'training_diagnostic', 'history',
    ):
        assert key in results, f"missing result key: {key}"
    assert results['total_updates'] == TOTAL_UPDATES
    assert len(results['history']['step']) == TOTAL_UPDATES
    assert 0.0 <= float(results['test_acc']) <= 1.0
    assert results['training_time_seconds'] > 0.0
    assert results['training_diagnostic']['n_logged_steps'] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])