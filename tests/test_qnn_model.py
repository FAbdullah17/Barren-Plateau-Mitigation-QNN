"""Tests for the QuantumNeuralNetwork Keras model.

Capacity/parameter-count assertions live in ``test_model_capacity.py``;
LayerwiseQNN invariant tests live in ``test_layerwise.py``.
"""

import pytest
import numpy as np
import cirq
import tensorflow as tf
import tensorflow_quantum as tfq

from src.models.qnn_model import QuantumNeuralNetwork


def _dummy_batch(n_qubits, batch_size=5):
    del n_qubits
    return tfq.convert_to_tensor([cirq.Circuit() for _ in range(batch_size)])


class TestQuantumNeuralNetwork:
    def test_is_keras_model(self):
        assert isinstance(QuantumNeuralNetwork(4, 2), tf.keras.Model)

    def test_forward_output_shape(self):
        model = QuantumNeuralNetwork(4, 2, cost="global")
        out = model(_dummy_batch(4), training=False)
        assert out.shape == (5, 1)

    def test_output_in_unit_range(self):
        model = QuantumNeuralNetwork(4, 2, cost="local")
        out = model(_dummy_batch(4), training=False).numpy()
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_invalid_cost_rejected(self):
        with pytest.raises(ValueError):
            QuantumNeuralNetwork(4, 2, cost="banana")

    def test_same_init_seed_reproduces_parameters(self):
        a = QuantumNeuralNetwork(4, 2, cost="global", init_seed=42)
        b = QuantumNeuralNetwork(4, 2, cost="global", init_seed=42)
        a._build_once()
        b._build_once()
        assert np.array_equal(
            a.trainable_variables[0].numpy(),
            b.trainable_variables[0].numpy(),
        )

    def test_different_init_seeds_give_different_parameters(self):
        a = QuantumNeuralNetwork(4, 2, cost="global", init_seed=42)
        b = QuantumNeuralNetwork(4, 2, cost="global", init_seed=123)
        a._build_once()
        b._build_once()
        assert not np.array_equal(
            a.trainable_variables[0].numpy(),
            b.trainable_variables[0].numpy(),
        )

    def test_get_num_parameters_works_before_forward_pass(self):
        model = QuantumNeuralNetwork(6, 4, cost="global")
        assert model.get_num_parameters() == 2 * 6 * 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
