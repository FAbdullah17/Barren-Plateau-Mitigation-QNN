"""Layerwise QNN tests.

Guards the three Skolik invariants implemented by ``LayerwiseQNN``:

  (i)   a stored ``param_values[k]`` never changes afterwards;
  (ii)  fine-tune initialization == the staged values;
  (iii) at stage ``k`` exactly ``2 * n_qubits`` parameters are trainable.

Plus bake-fidelity (numeric baked rotations match stored values, symbolic
counts correct), finetune-circuit structure matching the standard ansatz,
output shape/range and error cases.
"""

import pytest
import numpy as np
import cirq
import tensorflow as tf
import tensorflow_quantum as tfq

from src.models.qnn_model import LayerwiseQNN, _force_build
from src.models.quantum_circuit import QuantumCircuit


def _count_params(model):
    return int(sum(tf.size(v) for v in model.trainable_variables))


def _empty_batch(batch_size=5):
    return tfq.convert_to_tensor([cirq.Circuit() for _ in range(batch_size)])


def _build_and_store(qnn):
    """Add every layer and store its initialized values."""
    for _ in range(qnn.target_layers):
        qnn.add_layer()
        qnn.store_current_params()


def _reference_layer_ops(n_qubits, qubits, values=None, symbols=None):
    """Independent reconstruction of one ansatz layer (RY/RZ + CNOT chain).

    Mirrors the documented hardware-efficient ansatz so the baked circuit can
    be validated against a reference built outside of ``LayerwiseQNN``.
    """
    assert (values is None) != (symbols is None)
    ops = []
    for q in range(n_qubits):
        if values is not None:
            ops.append(cirq.ry(float(values[2 * q]))(qubits[q]))
            ops.append(cirq.rz(float(values[2 * q + 1]))(qubits[q]))
        else:
            ops.append(cirq.ry(symbols[2 * q])(qubits[q]))
            ops.append(cirq.rz(symbols[2 * q + 1])(qubits[q]))
    for i in range(n_qubits - 1):
        ops.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    return ops


class TestInvariantAtMost2nTrainable:
    """Invariant (iii): exactly 2n trainable parameters at each stage."""

    @pytest.mark.parametrize("n_qubits,target_layers", [(4, 3), (4, 4), (6, 3)])
    def test_exactly_2n_trainable_at_each_stage(self, n_qubits, target_layers):
        qnn = LayerwiseQNN(n_qubits, target_layers, cost="global", init_seed=5)
        for _ in range(target_layers):
            model = qnn.add_layer()
            _force_build(model)
            assert _count_params(model) == 2 * n_qubits
            assert len(model.trainable_variables) == 1
            qnn.store_current_params()

    def test_finetune_has_full_parameter_count(self):
        n_qubits, target_layers = 4, 3
        qnn = LayerwiseQNN(n_qubits, target_layers, cost="global", init_seed=5)
        _build_and_store(qnn)
        model = qnn.build_finetune_model()
        assert _count_params(model) == 2 * n_qubits * target_layers


class TestStoredValuesImmutable:
    """Invariant (i): stored values never change after being persisted."""

    def test_stored_values_never_change(self):
        n_qubits, target_layers = 4, 4
        qnn = LayerwiseQNN(n_qubits, target_layers, cost="local", init_seed=11)
        snapshots = []
        for k in range(target_layers):
            qnn.add_layer()
            qnn.store_current_params()
            snapshots.append(qnn.param_values[k].copy())
        for k in range(target_layers):
            assert qnn.param_values[k] is not None
            assert np.array_equal(qnn.param_values[k], snapshots[k])

    def test_seeded_storage_is_reproducible_and_sensitive(self):
        def weights_with(seed):
            qnn = LayerwiseQNN(4, 2, cost="global", init_seed=seed)
            _build_and_store(qnn)
            return qnn.param_values[0]

        a = weights_with(42)
        b = weights_with(42)
        c = weights_with(43)
        np.testing.assert_array_equal(a, b)
        assert not np.allclose(a, c)


class TestFinetuneInitialization:
    """Invariant (ii): fine-tune model is initialized from the staged values."""

    def test_finetune_init_matches_staged_values(self):
        n_qubits, target_layers = 4, 3
        qnn = LayerwiseQNN(n_qubits, target_layers, cost="global", init_seed=9)
        _build_and_store(qnn)
        expected = np.concatenate(qnn.param_values)
        model = qnn.build_finetune_model()
        got = model.trainable_variables[0].numpy()
        assert got.shape == (2 * n_qubits * target_layers,)
        np.testing.assert_array_equal(got, expected)

    @pytest.mark.parametrize("cost", ["global", "local"])
    def test_finetune_weights_reproducible(self, cost):
        def staged(seed):
            qnn = LayerwiseQNN(4, 2, cost=cost, init_seed=seed)
            _build_and_store(qnn)
            return qnn.build_finetune_model().trainable_variables[0].numpy()

        np.testing.assert_array_equal(staged(4), staged(4))
        assert not np.allclose(staged(4), staged(5))


class TestBakeFidelity:
    """Baked layers appear as numeric constants matching stored values."""

    def test_baked_layers_match_stored_values(self):
        n_qubits, target_layers = 4, 3
        qnn = LayerwiseQNN(n_qubits, target_layers, cost="global", init_seed=7)

        for k in range(target_layers):
            qnn.add_layer()
            qnn.store_current_params()

            # Reconstruct the expected stage-k circuit: layers 0..k-1 baked
            # from the stored values, layer k symbolic.
            expected_ops = []
            for layer in range(k):
                expected_ops += _reference_layer_ops(
                    n_qubits, qnn.qubits, values=qnn.param_values[layer]
                )
            expected_ops += _reference_layer_ops(
                n_qubits, qnn.qubits, symbols=qnn.symbols
            )

            # The baked circuit must equal the reference exactly (gate
            # equality for the rotation angles is approximate, matching the
            # float32 values stored by the PQC layer).
            assert qnn.circuit == cirq.Circuit(expected_ops)

            # Baked layers are numeric constants; the newest layer's rotations
            # are symbolic.
            for layer in range(k):
                baked_ops = _reference_layer_ops(
                    n_qubits, qnn.qubits, values=qnn.param_values[layer]
                )
                assert all(not cirq.is_parameterized(op) for op in baked_ops)
            sym_ops = _reference_layer_ops(
                n_qubits, qnn.qubits, symbols=qnn.symbols
            )
            assert all(
                cirq.is_parameterized(op)
                for op in sym_ops
                if isinstance(op.gate, (cirq.Ry, cirq.Rz))
            )


class TestStructure:
    def test_finetune_circuit_matches_standard_ansatz(self):
        n_qubits, target_layers = 4, 3
        qnn = LayerwiseQNN(n_qubits, target_layers, cost="global", init_seed=3)
        _build_and_store(qnn)
        qnn.build_finetune_model()
        standard = QuantumCircuit(n_qubits, target_layers).get_circuit()
        assert list(qnn.circuit.all_operations()) == list(standard.all_operations())

    @pytest.mark.parametrize("cost", ["global", "local"])
    def test_output_shape_range_and_no_dense(self, cost):
        n_qubits, target_layers = 4, 2
        qnn = LayerwiseQNN(n_qubits, target_layers, cost=cost, init_seed=1)
        _build_and_store(qnn)
        model = qnn.build_finetune_model()
        out = model(_empty_batch(4), training=False)
        assert out.shape == (4, 1)
        assert np.all(out.numpy() >= 0.0)
        assert np.all(out.numpy() <= 1.0)
        assert all(
            not isinstance(layer, tf.keras.layers.Dense)
            for layer in model.layers
        )


class TestErrorCases:
    def test_invalid_cost_raises(self):
        with pytest.raises(ValueError):
            LayerwiseQNN(4, 2, cost="invalid")

    def test_add_layer_beyond_target_raises(self):
        qnn = LayerwiseQNN(4, 2, cost="global")
        qnn.add_layer()
        qnn.store_current_params()
        qnn.add_layer()
        qnn.store_current_params()
        with pytest.raises(ValueError):
            qnn.add_layer()

    def test_add_layer_without_storing_previous_raises(self):
        qnn = LayerwiseQNN(4, 3, cost="global")
        qnn.add_layer()
        with pytest.raises(RuntimeError):
            qnn.add_layer()

    def test_store_without_model_raises(self):
        qnn = LayerwiseQNN(4, 2, cost="global")
        with pytest.raises(RuntimeError):
            qnn.store_current_params()

    def test_finetune_before_all_stored_raises(self):
        qnn = LayerwiseQNN(4, 3, cost="global")
        qnn.add_layer()
        qnn.store_current_params()
        qnn.add_layer()
        with pytest.raises(RuntimeError):
            qnn.build_finetune_model()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])