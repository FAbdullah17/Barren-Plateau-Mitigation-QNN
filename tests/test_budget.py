"""Training-budget tests.

Guarantees the equal-budget control:

* ``compute_layerwise_budget`` splits ``total_updates`` so the entries sum to
  the total and every phase receives at least one update.
* The baseline trainer consumes exactly ``total_updates`` gradient steps.
* The layerwise trainer consumes exactly ``total_updates`` across staged
  layers + fine-tune, and records the split in the results.
"""

import pytest
import numpy as np
import cirq
import tensorflow_quantum as tfq

from src.training.baseline_trainer import BaselineTrainer
from src.training.layerwise_trainer import LayerwiseTrainer, compute_layerwise_budget


def _random_circuits(n_samples, n_qubits, seed):
    rng = np.random.RandomState(seed)
    qubits = cirq.GridQubit.rect(1, n_qubits)
    circuits = []
    for _ in range(n_samples):
        circuit = cirq.Circuit()
        for q, angle in zip(qubits, rng.uniform(-np.pi, np.pi, size=n_qubits)):
            circuit.append(cirq.ry(angle)(q))
        circuits.append(circuit)
    return tfq.convert_to_tensor(circuits)


def _random_labels(n_samples, seed):
    return np.random.RandomState(seed).randint(0, 2, size=n_samples)


class TestBudgetSplit:
    @pytest.mark.parametrize("total_updates,n_layers", [
        (2500, 4), (2500, 3), (120, 2), (10, 1), (9, 2),
    ])
    def test_split_sums_to_total(self, total_updates, n_layers):
        split = compute_layerwise_budget(total_updates, n_layers)
        assert split['per_stage'] >= 1
        assert split['finetune'] >= 0
        assert split['per_stage'] * n_layers + split['finetune'] == total_updates

    def test_finetune_gets_remainder(self):
        split = compute_layerwise_budget(2500, 4)
        assert split['per_stage'] == 500
        assert split['finetune'] == 500

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            compute_layerwise_budget(0, 2)
        with pytest.raises(ValueError):
            compute_layerwise_budget(10.5, 2)
        with pytest.raises(ValueError):
            compute_layerwise_budget(10, 0)
        # total_updates too small for every phase to get at least one step
        with pytest.raises(ValueError):
            compute_layerwise_budget(2, 4)


class TestBaselineBudget:
    def test_consumes_exactly_total_updates(self):
        n_qubits, n_layers = 2, 1
        total_updates = 6
        train_circuits = _random_circuits(8, n_qubits, seed=1)
        train_labels = _random_labels(8, seed=2)
        val_circuits = _random_circuits(4, n_qubits, seed=3)
        val_labels = _random_labels(4, seed=4)

        trainer = BaselineTrainer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            cost='global',
            learning_rate=0.01,
            batch_size=4,
            total_updates=total_updates,
            log_frequency=3,
            diagnostic_samples=4,
            init_seed=11,
            training_seed=12,
        )
        results = trainer.train(
            train_circuits=train_circuits,
            train_labels=train_labels,
            val_circuits=val_circuits,
            val_labels=val_labels,
        )

        assert results['total_updates'] == total_updates
        assert len(results['history']['step']) == total_updates
        assert results['history']['step'] == list(range(total_updates))
        assert results['layerwise_budget_split'] is None
        # Runtime-derived parameter count.
        assert results['n_parameters'] == 2 * n_qubits * n_layers
        assert results['training_diagnostic']['n_parameters'] == 2 * n_qubits * n_layers
        assert results['training_diagnostic']['n_logged_steps'] >= 1
        assert 0.0 <= results['test_acc'] <= 1.0


class TestLayerwiseBudget:
    def test_consumes_exactly_total_updates_and_records_split(self):
        n_qubits, n_layers = 2, 2
        total_updates = 9
        train_circuits = _random_circuits(8, n_qubits, seed=1)
        train_labels = _random_labels(8, seed=2)
        val_circuits = _random_circuits(4, n_qubits, seed=3)
        val_labels = _random_labels(4, seed=4)

        trainer = LayerwiseTrainer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            cost='global',
            learning_rate=0.01,
            batch_size=4,
            total_updates=total_updates,
            log_frequency=3,
            diagnostic_samples=4,
            init_seed=11,
            training_seed=12,
        )
        results = trainer.train(
            train_circuits=train_circuits,
            train_labels=train_labels,
            val_circuits=val_circuits,
            val_labels=val_labels,
        )

        split = results['layerwise_budget_split']
        assert split == {'per_stage': 3, 'finetune': 3}
        assert split['per_stage'] * n_layers + split['finetune'] == total_updates
        assert results['total_updates'] == total_updates
        assert len(results['history']['step']) == total_updates
        assert results['history']['step'] == list(range(total_updates))
        # Final model is the fine-tuned full ansatz (runtime-derived count).
        assert results['n_parameters'] == 2 * n_qubits * n_layers
        assert results['training_diagnostic']['n_parameters'] == 2 * n_qubits * n_layers
        assert 0.0 <= results['test_acc'] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])