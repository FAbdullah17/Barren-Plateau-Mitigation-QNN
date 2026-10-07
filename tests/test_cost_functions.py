"""Tests for the cost readout operators.

Covers: global readout support on all qubits; local readout as a scaled sum;
both produce a ``(batch, 1)`` output through the same PQC; outputs in [0, 1].
"""

import pytest
import cirq
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq

from src.models.quantum_circuit import create_readout_operators
from src.models.qnn_model import QuantumNeuralNetwork


def _qubits(n_qubits):
    return set(cirq.GridQubit.rect(1, n_qubits))


def _dummy_batch(n_qubits, batch_size=5):
    del n_qubits  # empty circuits are valid PQC inputs
    return tfq.convert_to_tensor([cirq.Circuit() for _ in range(batch_size)])


def _data_batch(n_qubits, batch_size=20):
    """Batch of distinct data-encoded circuits (ry(pi * x_i) per qubit)."""
    qubits = cirq.GridQubit.rect(1, n_qubits)
    rng = np.random.default_rng(0)
    circuits = []
    for _ in range(batch_size):
        circuit = cirq.Circuit()
        for qubit in qubits:
            angle = float(rng.uniform(0.0, np.pi))
            circuit.append(cirq.ry(angle)(qubit))
        circuits.append(circuit)
    return tfq.convert_to_tensor(circuits)


class TestReadoutOperators:
    def test_global_operator_has_support_on_all_qubits(self):
        n_qubits = 6
        obs = create_readout_operators(n_qubits, local=False)
        terms = list(obs)
        assert len(terms) == 1
        assert set(terms[0].qubits) == _qubits(n_qubits)
        assert abs(terms[0].coefficient) == pytest.approx(1.0)

    def test_local_operator_is_scaled_sum(self):
        n_qubits = 4
        obs = create_readout_operators(n_qubits, local=True)
        terms = list(obs)
        assert len(terms) == n_qubits
        for term in terms:
            assert len(term.qubits) == 1
            assert abs(term.coefficient) == pytest.approx(1.0 / n_qubits)

    def test_local_coefficients_sum_to_one(self):
        n_qubits = 8
        obs = create_readout_operators(n_qubits, local=True)
        assert sum(term.coefficient for term in obs) == pytest.approx(1.0)


class TestPQCThroughModel:
    @pytest.mark.parametrize("cost", ["global", "local"])
    def test_output_shape_is_batch_one(self, cost):
        model = QuantumNeuralNetwork(n_qubits=4, n_layers=2, cost=cost)
        out = model(_dummy_batch(4), training=False)
        assert out.shape == (5, 1)

    @pytest.mark.parametrize("cost", ["global", "local"])
    def test_output_in_unit_range(self, cost):
        model = QuantumNeuralNetwork(n_qubits=4, n_layers=2, cost=cost)
        out = model(_dummy_batch(4), training=False).numpy()
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_global_output_varies_across_data(self):
        # The genuinely-global observable must respond to the input data:
        # distinct data-encoded circuits should give distinct outputs.
        model = QuantumNeuralNetwork(n_qubits=4, n_layers=2, cost="global", init_seed=1)
        out = model(_data_batch(4, batch_size=20), training=False).numpy()
        assert out.std() > 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
