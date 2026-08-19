"""Step-based baseline training.

Trains the full parameterized quantum circuit end to end with exactly
``total_updates`` gradient steps — the same budget that layerwise training
consumes (split across its stages). The dataset is shuffled once with
``training_seed`` (decoupled from ``data_seed``/``init_seed``).

Every ``log_frequency`` steps the per-sample Jacobian ``(B, P)`` is computed
on ``diagnostic_samples`` fixed training samples and fed to ``GradientTracker``
(the training diagnostic). Results use the metrics schema.

When ``checkpoint_dir`` is set, the model weights, Adam state, step counter and
diagnostic accumulation are saved every ``checkpoint_frequency`` steps and a
later call with the same seeds resumes exactly from the latest checkpoint.
"""

import os
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')  # TFQ requires Keras 2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import tensorflow as tf
import numpy as np
import time
import json
from typing import Dict, Optional

from src.models import QuantumNeuralNetwork
from src.evaluation.metrics import GradientTracker


class BaselineTrainer:
    """Standard end-to-end training for quantum neural networks."""

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 4,
        cost: str = 'global',
        learning_rate: float = 0.01,
        batch_size: int = 20,
        total_updates: int = 2500,
        log_frequency: int = 10,
        diagnostic_samples: int = 100,
        init_seed: Optional[int] = None,
        training_seed: Optional[int] = None,
        track_gradients: bool = True,
        checkpoint_dir: Optional[str] = None,
        checkpoint_frequency: int = 500,
    ):
        """
        Initialize the step-based baseline trainer.

        Args:
            n_qubits: Number of qubits.
            n_layers: Number of circuit layers.
            cost: 'global' or 'local' cost function.
            learning_rate: Adam learning rate.
            batch_size: Mini-batch size.
            total_updates: Total number of gradient steps.
            log_frequency: Compute/log the gradient diagnostic every N steps.
            diagnostic_samples: Number of fixed training samples used for the
                per-sample Jacobian (accumulate >= 100 samples per logged step
                to stabilize the variance estimate).
            init_seed: PQC parameter init seed (decoupled from training).
            training_seed: Dataset shuffle seed (decoupled from data/init).
            track_gradients: Compute the gradient diagnostic when True.
            checkpoint_dir: Directory for resumable checkpoints (model
                weights, optimizer state, step counter, diagnostic state).
                When None, checkpoints are disabled.
            checkpoint_frequency: Save a checkpoint every N gradient steps.
        """
        if cost not in ('global', 'local'):
            raise ValueError(f"cost must be 'global' or 'local', got {cost!r}")
        if total_updates < 1:
            raise ValueError(f"total_updates must be >= 1, got {total_updates}")
        if log_frequency < 1:
            raise ValueError(f"log_frequency must be >= 1, got {log_frequency}")
        if diagnostic_samples < 1:
            raise ValueError(
                f"diagnostic_samples must be >= 1, got {diagnostic_samples}"
            )
        if checkpoint_frequency < 0:
            raise ValueError(
                f"checkpoint_frequency must be >= 0, got {checkpoint_frequency}"
            )

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.cost = cost
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.total_updates = total_updates
        self.log_frequency = log_frequency
        self.diagnostic_samples = diagnostic_samples
        self.init_seed = init_seed
        self.training_seed = training_seed
        self.track_gradients = track_gradients
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency

        if training_seed is not None:
            tf.random.set_seed(training_seed)
            np.random.seed(training_seed)

        self.model = QuantumNeuralNetwork(
            n_qubits=n_qubits,
            n_layers=n_layers,
            cost=cost,
            init_seed=init_seed,
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.per_sample_loss = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.gradient_tracker = GradientTracker()

        # Runtime-derived parameter count.
        self.n_parameters = self.model.get_num_parameters()

        self._diag_circuits = None
        self._diag_labels = None

        # Resumable checkpoint state (model + optimizer + step + diagnostic).
        self._ckpt = None
        self._step_var = None
        if checkpoint_dir and checkpoint_frequency > 0:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            self._step_var = tf.Variable(0, dtype=tf.int64)
            self._ckpt = tf.train.Checkpoint(
                model=self.model, optimizer=self.optimizer, step=self._step_var
            )

    def train(
        self,
        train_circuits: tf.Tensor,
        train_labels: np.ndarray,
        val_circuits: tf.Tensor,
        val_labels: np.ndarray,
    ) -> Dict:
        """
        Train for exactly ``total_updates`` gradient steps.

        Args:
            train_circuits: Training quantum circuits.
            train_labels: Training labels.
            val_circuits: Validation/test quantum circuits.
            val_labels: Validation/test labels.

        Returns:
            Results dict following the metrics schema.
        """
        n_train = len(train_labels)
        if n_train < 1:
            raise ValueError("train_labels must not be empty")

        # Resume from the latest checkpoint if one exists.
        completed = self._restore_checkpoint()

        # Dataset shuffled once with training_seed, then repeated and batched.
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_circuits, train_labels)
        )
        train_dataset = train_dataset.shuffle(
            n_train,
            seed=self.training_seed,
            reshuffle_each_iteration=False,
        ).repeat().batch(self.batch_size)
        if completed > 0:
            train_dataset = train_dataset.skip(completed)
        train_dataset = train_dataset.take(self.total_updates - completed)

        # Fixed diagnostic samples (stable Jacobian estimate).
        self._diag_circuits = train_circuits[: self.diagnostic_samples]
        self._diag_labels = train_labels[: self.diagnostic_samples]

        history = {
            'step': [],
            'train_loss': [],
            'train_acc': [],
            'val_step': [],
            'val_loss': [],
            'val_acc': [],
        }

        start_time = time.time()
        for i, (batch_circuits, batch_labels) in enumerate(train_dataset):
            step = completed + i
            loss, acc = self._train_step(batch_circuits, batch_labels)
            history['step'].append(step)
            history['train_loss'].append(float(loss))
            history['train_acc'].append(float(acc))

            if (step + 1) % self.log_frequency == 0 or (step + 1) == self.total_updates:
                val_loss, val_acc = self._evaluate(val_circuits, val_labels)
                history['val_step'].append(step + 1)
                history['val_loss'].append(float(val_loss))
                history['val_acc'].append(float(val_acc))
                if self.track_gradients:
                    self._log_diagnostic(step + 1)

            if (
                self._ckpt is not None
                and (step + 1) % self.checkpoint_frequency == 0
            ):
                self._save_checkpoint(step + 1)

        training_time = time.time() - start_time

        test_loss, test_acc = self._evaluate(val_circuits, val_labels)

        diagnostic = self.gradient_tracker.get_statistics()
        diagnostic['n_parameters'] = self.n_parameters

        return {
            'config': {
                'approach': 'baseline',
                'n_qubits': self.n_qubits,
                'n_layers': self.n_layers,
                'cost': self.cost,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'total_updates': self.total_updates,
                'log_frequency': self.log_frequency,
                'init_seed': self.init_seed,
                'training_seed': self.training_seed,
                'track_gradients': self.track_gradients,
            },
            'total_updates': self.total_updates,
            'layerwise_budget_split': None,
            'n_parameters': self.n_parameters,
            'test_loss': float(test_loss),
            'test_acc': float(test_acc),
            'training_time_seconds': training_time,
            'training_diagnostic': diagnostic,
            'history': history,
        }

    def _train_step(self, circuits, labels):
        """Single gradient step; returns (loss, accuracy) as floats."""
        with tf.GradientTape() as tape:
            predictions = self.model(circuits, training=True)
            predictions = tf.squeeze(predictions, axis=-1)
            loss = self.loss_fn(labels, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        accuracy = self._batch_accuracy(predictions, labels)
        return float(loss.numpy()), float(accuracy.numpy())

    def _log_diagnostic(self, step: int) -> None:
        """Per-sample Jacobian (B, P) on fixed samples -> gradient tracker.

        The per-sample loss must be computed from ``(B, 1)`` inputs: legacy
        Keras ``BinaryCrossentropy(reduction=NONE)`` returns per-sample losses
        only for 2-D inputs. The resulting Jacobian is ``(B, P)``.
        """
        labels_2d = tf.cast(self._diag_labels, tf.float32)[:, None]
        with tf.GradientTape() as tape:
            predictions = self.model(self._diag_circuits, training=False)
            loss_per_sample = self.per_sample_loss(labels_2d, predictions)
        jacobian = tape.jacobian(
            loss_per_sample, self.model.trainable_variables[0]
        )
        self.gradient_tracker.update(
            jacobian.numpy(), step=step, samples=self._diag_circuits.shape[0]
        )

    def _evaluate(self, circuits, labels):
        """Return (loss, accuracy) on the given data."""
        predictions = self.model(circuits, training=False)
        predictions = tf.squeeze(predictions, axis=-1)
        loss = self.loss_fn(labels, predictions)
        accuracy = self._batch_accuracy(predictions, labels)
        return float(loss.numpy()), float(accuracy.numpy())

    def _restore_checkpoint(self) -> int:
        """Restore the latest checkpoint; returns completed gradient steps.

        Restores model weights, Adam state, the step counter, and the
        diagnostic accumulation so a resumed run is identical to a
        never-interrupted one.
        """
        if self._ckpt is None:
            return 0
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest is None:
            print(f"No checkpoint found in {self.checkpoint_dir}; starting fresh")
            return 0
        self._ckpt.restore(latest)
        completed = int(self._step_var.numpy())
        self._restore_tracker()
        print(
            f"Resuming from step {completed} of {self.total_updates} "
            f"(checkpoint {latest})"
        )
        return completed

    def _save_checkpoint(self, completed_steps: int) -> None:
        """Persist model + optimizer + step + diagnostic accumulation."""
        self._step_var.assign(completed_steps)
        prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self._ckpt.save(file_prefix=prefix)
        self._save_tracker()
        print(f"Checkpoint saved at step {completed_steps}")

    def _save_tracker(self) -> None:
        """Persist the diagnostic accumulation alongside the checkpoint."""
        state = self.gradient_tracker.state_dict()
        tmp = os.path.join(self.checkpoint_dir, 'tracker.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(state, f)
        os.replace(tmp, os.path.join(self.checkpoint_dir, 'tracker.json'))

    def _restore_tracker(self) -> None:
        """Reload the diagnostic accumulation (no-op if absent)."""
        path = os.path.join(self.checkpoint_dir, 'tracker.json')
        if not os.path.exists(path):
            return
        with open(path, 'r') as f:
            self.gradient_tracker.restore_state(json.load(f))

    @staticmethod
    def _batch_accuracy(predictions, labels):
        preds_binary = tf.cast(predictions > 0.5, tf.int32)
        labels_int = tf.cast(labels, tf.int32)
        return tf.reduce_mean(
            tf.cast(tf.equal(preds_binary, labels_int), tf.float32)
        )