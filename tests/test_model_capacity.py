"""Capacity tests.

Structural identity of global vs local models; zero trainable classical
layers; runtime-derived parameter counts read from the instantiated model
(never from config or manual arithmetic).
"""

import pytest
import numpy as np
import tensorflow as tf
import cirq
import tensorflow_quantum as tfq

from src.models.qnn_model import QuantumNeuralNetwork


def _dummy_batch(n_qubits, batch_size=5):
    del n_qubits
    return tfq.convert_to_tensor([cirq.Circuit() for _ in range(batch_size)])


def _count_params(model):
    return int(sum(tf.size(v) for v in model.trainable_variables))


class TestStructuralIdentity:
    @pytest.mark.parametrize("n_qubits,n_layers", [(4, 2), (4, 3), (6, 4)])
    def test_global_and_local_have_identical_parameter_count(
        self, n_qubits, n_layers
    ):
        g = QuantumNeuralNetwork(n_qubits, n_layers, cost="global")
        l = QuantumNeuralNetwork(n_qubits, n_layers, cost="local")
        assert _count_params(g) == _count_params(l) == 2 * n_qubits * n_layers

    def test_zero_trainable_classical_layers(self):
        model = QuantumNeuralNetwork(4, 3, cost="global")
        model(_dummy_batch(4), training=False)
        assert all(
            not isinstance(layer, tf.keras.layers.Dense)
            for layer in model.layers
        )
        # Only the single PQC parameter vector is trainable.
        assert len(model.trainable_variables) == 1

    def test_fixed_output_head(self):
        model = QuantumNeuralNetwork(4, 2, cost="global")
        assert isinstance(model.output_layer, tf.keras.layers.Lambda)

    def test_same_circuit_for_both_costs(self):
        g = QuantumNeuralNetwork(4, 2, cost="global")
        l = QuantumNeuralNetwork(4, 2, cost="local")
        assert g.circuit == l.circuit


class TestRuntimeParameterCount:
    @pytest.mark.parametrize("n_qubits,n_layers", [(4, 2), (4, 4), (6, 2), (8, 3)])
    def test_n_parameters_matches_instantiated_model(self, n_qubits, n_layers):
        model = QuantumNeuralNetwork(n_qubits, n_layers, cost="global")
        assert model.get_num_parameters() == 2 * n_qubits * n_layers
        # Same value whether read via the getter or directly from variables.
        model._build_once()
        assert model.get_num_parameters() == _count_params(model)

    def test_parameter_vector_shape(self):
        n_qubits, n_layers = 4, 3
        model = QuantumNeuralNetwork(n_qubits, n_layers, cost="local")
        model._build_once()
        weights = model.trainable_variables[0].numpy()
        assert weights.shape == (2 * n_qubits * n_layers,)
        assert np.all(np.isfinite(weights))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
