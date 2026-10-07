"""Step-based layerwise training.

Incremental Skolik training: each new ansatz layer is trained with the earlier
layers baked in as frozen numeric constants (``LayerwiseQNN``), then persisted
with ``store_current_params``; finally the full ansatz is fine-tuned starting
exactly from the staged values (``build_finetune_model``).

The trainer consumes exactly ``total_updates`` gradient steps, split as
``per_stage = total_updates // (n_layers + 1)`` for each staged layer and the
remainder for fine-tuning — the same total budget as the baseline, so any
difference is *how* the updates are distributed, not how many. The dataset is
shuffled once with ``training_seed``; ``init_seed`` (decoupled) drives the
per-layer parameter initializer.
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

from src.models import LayerwiseQNN
from src.evaluation.metrics import GradientTracker


def compute_layerwise_budget(total_updates: int, n_layers: int) -> Dict:
    """Split ``total_updates`` across staged layers and the fine-tune phase.

    ``per_stage = total_updates // (n_layers + 1)``; fine-tuning receives the
    remainder. The two entries always sum to ``total_updates``.

    Returns:
        dict with ``per_stage`` and ``finetune``.

    Raises:
        ValueError: if either input is invalid or ``total_updates`` is too
            small for every phase to receive at least one update.
    """
    if not isinstance(total_updates, int) or total_updates < 1:
        raise ValueError(f"total_updates must be a positive integer, got {total_updates}")
    if not isinstance(n_layers, int) or n_layers < 1:
        raise ValueError(f"n_layers must be a positive integer, got {n_layers}")

    n_phases = n_layers + 1  # staged layers + fine-tune
    per_stage = total_updates // n_phases
    if per_stage < 1:
        raise ValueError(
            f"total_updates ({total_updates}) is too small for {n_layers} layers "
            f"+ fine-tune"
        )
    finetune = total_updates - per_stage * n_layers
    return {'per_stage': int(per_stage), 'finetune': int(finetune)}


class LayerwiseTrainer:
    """Incremental Skolik training with a fixed step budget."""

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
        Initialize the step-based layerwise trainer.

        Args:
            n_qubits: Number of qubits.
            n_layers: Target number of layers (also used for the budget split).
            cost: 'global' or 'local' cost function.
            learning_rate: Adam learning rate.
            batch_size: Mini-batch size.
            total_updates: Total gradient steps across all phases.
            log_frequency: Compute/log the gradient diagnostic every N steps.
            diagnostic_samples: Fixed training samples for the per-sample
                Jacobian.
            init_seed: Base seed for per-layer parameter initialization.
            training_seed: Dataset shuffle seed (decoupled).
            track_gradients: Compute the gradient diagnostic when True.
            checkpoint_dir: Directory for resumable phase checkpoints. Staged
                parameter values and the diagnostic accumulation are persisted
                after every phase (each phase is short), so an interrupted run
                resumes at the last phase boundary. ``checkpoint_frequency`` is
                accepted for API consistency but not used here.
            checkpoint_frequency: Ignored (see ``checkpoint_dir``).
        """
        if cost not in ('global', 'local'):
            raise ValueError(f"cost must be 'global' or 'local', got {cost!r}")
        if log_frequency < 1:
            raise ValueError(f"log_frequency must be >= 1, got {log_frequency}")
        if diagnostic_samples < 1:
            raise ValueError(
                f"diagnostic_samples must be >= 1, got {diagnostic_samples}"
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

        self.budget = compute_layerwise_budget(total_updates, n_layers)

        if training_seed is not None:
            tf.random.set_seed(training_seed)
            np.random.seed(training_seed)

        self.qnn = LayerwiseQNN(
            n_qubits=n_qubits,
            target_layers=n_layers,
            cost=cost,
            init_seed=init_seed,
        )
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.per_sample_loss = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.gradient_tracker = GradientTracker()

        self._diag_circuits = None
        self._diag_labels = None
        self._val_circuits = None
        self._val_labels = None

    def train(
        self,
        train_circuits: tf.Tensor,
        train_labels: np.ndarray,
        val_circuits: tf.Tensor,
        val_labels: np.ndarray,
    ) -> Dict:
        """
        Train staged layers then fine-tune, consuming exactly ``total_updates``.

        Args:
            train_circuits: Training quantum circuits.
            train_labels: Training labels.
            val_circuits: Validation/test quantum circuits.
            val_labels: Validation/test labels.

        Returns:
            Results dict following the metrics schema, including the
            recorded ``layerwise_budget_split``.
        """
        n_train = len(train_labels)
        if n_train < 1:
            raise ValueError("train_labels must not be empty")

        self._val_circuits = val_circuits
        self._val_labels = val_labels
        self._diag_circuits = train_circuits[: self.diagnostic_samples]
        self._diag_labels = train_labels[: self.diagnostic_samples]

        # Resume at the last phase boundary (staged values + diagnostic).
        start_layer, resumed = self._restore_phase_state()
        if resumed:
            print(
                f"Resuming layerwise training from layer {start_layer} "
                f"of {self.n_layers}"
            )

        per_stage = self.budget['per_stage']
        completed_steps = start_layer * per_stage

        # Shared, once-shuffled dataset. Each phase creates its own iterator
        # (the shuffle buffer re-initializes per phase), so no positional skip
        # is applied on resume: phases always read the same batches, matching
        # an uninterrupted run exactly.
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_circuits, train_labels)
        )
        train_dataset = train_dataset.shuffle(
            n_train,
            seed=self.training_seed,
            reshuffle_each_iteration=False,
        ).repeat().batch(self.batch_size)

        history = {
            'step': [],
            'train_loss': [],
            'train_acc': [],
            'val_step': [],
            'val_loss': [],
            'val_acc': [],
        }

        start_time = time.time()
        step = completed_steps

        # Staged phase: one layer at a time, earlier layers baked in and frozen.
        for _ in range(start_layer, self.n_layers):
            model = self.qnn.add_layer()
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            self._run_steps(
                model, optimizer, train_dataset, per_stage, history, step
            )
            step += per_stage
            self.qnn.store_current_params()
            self._save_phase_state()
            self._save_tracker()

        # Fine-tune phase: full ansatz initialized from the staged values.
        finetune_updates = self.budget['finetune']
        if finetune_updates > 0:
            model = self.qnn.build_finetune_model()
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            self._run_steps(
                model, optimizer, train_dataset, finetune_updates, history, step
            )
            step += finetune_updates

        training_time = time.time() - start_time

        model = self.qnn.get_current_model()
        test_loss, test_acc = self._evaluate(model, val_circuits, val_labels)

        # Runtime-derived parameter count of the final (fine-tune) model.
        n_parameters = int(np.prod(model.trainable_variables[0].shape))
        diagnostic = self.gradient_tracker.get_statistics()
        diagnostic['n_parameters'] = n_parameters

        return {
            'config': {
                'approach': 'layerwise',
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
            'layerwise_budget_split': dict(self.budget),
            'n_parameters': n_parameters,
            'test_loss': float(test_loss),
            'test_acc': float(test_acc),
            'training_time_seconds': training_time,
            'training_diagnostic': diagnostic,
            'history': history,
        }

    def _run_steps(
        self,
        model,
        optimizer,
        dataset,
        n_steps: int,
        history: Dict,
        start_step: int,
    ) -> None:
        """Run ``n_steps`` gradient steps on ``model``, appending history."""
        for i, (batch_circuits, batch_labels) in enumerate(dataset.take(n_steps)):
            loss, acc = self._train_step(model, optimizer, batch_circuits, batch_labels)
            step = start_step + i
            history['step'].append(step)
            history['train_loss'].append(float(loss))
            history['train_acc'].append(float(acc))

            if (step + 1) % self.log_frequency == 0 or (i + 1) == n_steps:
                val_loss, val_acc = self._evaluate(
                    model, self._val_circuits, self._val_labels
                )
                history['val_step'].append(step + 1)
                history['val_loss'].append(float(val_loss))
                history['val_acc'].append(float(val_acc))
                if self.track_gradients:
                    self._log_diagnostic(model, step + 1)

    def _train_step(self, model, optimizer, circuits, labels):
        """Single gradient step on ``model``; returns (loss, accuracy) floats."""
        with tf.GradientTape() as tape:
            predictions = model(circuits, training=True)
            predictions = tf.squeeze(predictions, axis=-1)
            loss = self.loss_fn(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        accuracy = self._batch_accuracy(predictions, labels)
        return float(loss.numpy()), float(accuracy.numpy())

    def _log_diagnostic(self, model, step: int) -> None:
        """Per-sample Jacobian (B, P) on fixed samples -> gradient tracker.

        The per-sample loss must be computed from ``(B, 1)`` inputs: legacy
        Keras ``BinaryCrossentropy(reduction=NONE)`` returns per-sample losses
        only for 2-D inputs. The resulting Jacobian is ``(B, P)``.
        """
        labels_2d = tf.cast(self._diag_labels, tf.float32)[:, None]
        with tf.GradientTape() as tape:
            predictions = model(self._diag_circuits, training=False)
            loss_per_sample = self.per_sample_loss(labels_2d, predictions)
        jacobian = tape.jacobian(loss_per_sample, model.trainable_variables[0])
        self.gradient_tracker.update(
            jacobian.numpy(), step=step, samples=self._diag_circuits.shape[0]
        )

    def _save_phase_state(self) -> None:
        """Persist staged parameter values and completed-layer count."""
        if self.checkpoint_dir is None:
            return
        state_path = Path(self.checkpoint_dir) / 'phase_state.json'
        state_path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {
            f'layer_{k}': self.qnn.param_values[k]
            for k in range(self.qnn.current_layers)
            if self.qnn.param_values[k] is not None
        }
        np.savez(os.path.join(self.checkpoint_dir, 'param_values.npz'), **arrays)

        payload = {'completed_layers': self.qnn.current_layers}
        tmp = state_path.with_suffix('.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(payload, f)
        os.replace(tmp, state_path)

    def _restore_phase_state(self) -> tuple:
        """Reload staged values at the last phase boundary.

        Returns:
            ``(start_layer, resumed)`` — the layer index to continue from and
            whether a saved state was found.
        """
        if self.checkpoint_dir is None:
            return 0, False
        state_path = Path(self.checkpoint_dir) / 'phase_state.json'
        param_path = Path(self.checkpoint_dir) / 'param_values.npz'
        if not state_path.exists() or not param_path.exists():
            return 0, False

        with open(state_path, 'r') as f:
            payload = json.load(f)
        completed = int(payload['completed_layers'])

        loaded = np.load(param_path)
        for k in range(completed):
            self.qnn.param_values[k] = np.asarray(
                loaded[f'layer_{k}'], dtype=np.float32
            )
        self.qnn.current_layers = completed

        tracker_path = Path(self.checkpoint_dir) / 'tracker.json'
        if tracker_path.exists():
            with open(tracker_path, 'r') as f:
                self.gradient_tracker.restore_state(json.load(f))
        return completed, True

    def _save_tracker(self) -> None:
        """Persist the diagnostic accumulation at each phase boundary."""
        if self.checkpoint_dir is None:
            return
        state = self.gradient_tracker.state_dict()
        path = Path(self.checkpoint_dir) / 'tracker.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(state, f)
        os.replace(tmp, path)

    def _evaluate(self, model, circuits, labels):
        """Return (loss, accuracy) on the given data."""
        predictions = model(circuits, training=False)
        predictions = tf.squeeze(predictions, axis=-1)
        loss = self.loss_fn(labels, predictions)
        accuracy = self._batch_accuracy(predictions, labels)
        return float(loss.numpy()), float(accuracy.numpy())

    @staticmethod
    def _batch_accuracy(predictions, labels):
        preds_binary = tf.cast(predictions > 0.5, tf.int32)
        labels_int = tf.cast(labels, tf.int32)
        return tf.reduce_mean(
            tf.cast(tf.equal(preds_binary, labels_int), tf.float32)
        )