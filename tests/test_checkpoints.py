"""Checkpoint/resume tests for the trainers.

Verifies that interrupting and resuming a run produces identical final
statistics to a never-interrupted run, for both the step-level baseline
checkpoints and the phase-boundary layerwise checkpoints.
"""

import numpy as np
import cirq
import tensorflow_quantum as tfq

from src.training import BaselineTrainer, LayerwiseTrainer


def _make_data(n, n_qubits=2, seed=0):
    rng = np.random.RandomState(seed)
    angles = rng.uniform(0, np.pi, size=(n, n_qubits))
    qubits = cirq.GridQubit.rect(1, n_qubits)
    circuits = []
    for row in angles:
        c = cirq.Circuit()
        for q, a in zip(qubits, row):
            c.append(cirq.ry(a)(q))
        circuits.append(c)
    labels = (np.arange(n) % 2).astype(np.float32)
    return tfq.convert_to_tensor(circuits), labels


class TestBaselineResume:
    def test_resumed_run_matches_reference(self, tmp_path):
        train_circuits, y_train = _make_data(12, seed=0)
        test_circuits, y_test = _make_data(4, seed=1)

        kwargs = dict(
            n_qubits=2, n_layers=1, cost='global', learning_rate=0.05,
            batch_size=4, log_frequency=3,
            diagnostic_samples=8, init_seed=1, training_seed=2,
            track_gradients=True,
        )

        # Reference: uninterrupted full run, no checkpoints.
        ref = BaselineTrainer(**kwargs, total_updates=12).train(
            train_circuits, y_train, test_circuits, y_test
        )

        # Interrupted run: first segment saves a checkpoint, second resumes.
        ckpt_dir = str(tmp_path / 'ckpt')
        first = BaselineTrainer(
            **kwargs, total_updates=6, checkpoint_dir=ckpt_dir,
            checkpoint_frequency=5,
        ).train(train_circuits, y_train, test_circuits, y_test)
        assert (tmp_path / 'ckpt' / 'tracker.json').exists()

        resumed = BaselineTrainer(
            **kwargs, total_updates=12, checkpoint_dir=ckpt_dir,
            checkpoint_frequency=5,
        ).train(train_circuits, y_train, test_circuits, y_test)

        assert resumed['test_acc'] == ref['test_acc']
        assert resumed['test_loss'] == ref['test_loss']
        assert resumed['training_diagnostic']['n_logged_steps'] == 4
        assert resumed['training_diagnostic']['trajectory']['step'] == [3, 6, 9, 12]
        np.testing.assert_allclose(
            resumed['training_diagnostic']['mean_param_grad_variance'],
            ref['training_diagnostic']['mean_param_grad_variance'],
        )

    def test_no_checkpoint_starts_fresh(self, tmp_path):
        train_circuits, y_train = _make_data(12, seed=0)
        test_circuits, y_test = _make_data(4, seed=1)

        trainer = BaselineTrainer(
            n_qubits=2, n_layers=1, cost='global', learning_rate=0.05,
            batch_size=4, total_updates=12, log_frequency=3,
            diagnostic_samples=8, init_seed=1, training_seed=2,
            checkpoint_dir=str(tmp_path / 'empty_ckpt'),
            checkpoint_frequency=5,
        )
        results = trainer.train(train_circuits, y_train, test_circuits, y_test)
        assert results['training_diagnostic']['n_logged_steps'] == 4
        assert len(results['history']['step']) == 12


class TestLayerwiseResume:
    def test_resumed_run_matches_reference(self, tmp_path):
        train_circuits, y_train = _make_data(12, seed=0)
        test_circuits, y_test = _make_data(4, seed=1)

        kwargs = dict(
            n_qubits=2, n_layers=2, cost='global', learning_rate=0.05,
            batch_size=4, total_updates=14, log_frequency=7,
            diagnostic_samples=8, init_seed=1, training_seed=2,
            track_gradients=True,
        )

        ref = LayerwiseTrainer(**kwargs).train(
            train_circuits, y_train, test_circuits, y_test
        )

        ckpt_dir = str(tmp_path / 'lw_ckpt')
        LayerwiseTrainer(**kwargs, checkpoint_dir=ckpt_dir).train(
            train_circuits, y_train, test_circuits, y_test
        )
        assert (tmp_path / 'lw_ckpt' / 'phase_state.json').exists()

        resumed = LayerwiseTrainer(**kwargs, checkpoint_dir=ckpt_dir).train(
            train_circuits, y_train, test_circuits, y_test
        )

        assert resumed['test_acc'] == ref['test_acc']
        assert resumed['test_loss'] == ref['test_loss']
        assert resumed['training_diagnostic']['n_logged_steps'] == 4
        np.testing.assert_allclose(
            resumed['training_diagnostic']['mean_param_grad_variance'],
            ref['training_diagnostic']['mean_param_grad_variance'],
        )