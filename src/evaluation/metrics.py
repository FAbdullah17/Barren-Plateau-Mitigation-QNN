"""Metrics and gradient-tracking utilities.

Two distinct gradient statistics are tracked, and never conflated:

* **Landscape statistic** ``V̄``: variance of the cost-function gradient over
  *random parameter draws* (``grad_instances: (R, P)``). This is the
  barren-plateau signature, estimated with a standard error and a bootstrap CI
  over parameters.
* **Training diagnostic** ``V̄^x``: per-parameter gradient variance over
  *samples within a batch* (``grad_matrix: (B, P)``) during training, recorded
  as a per-step trajectory.

Statistics are reported with their trajectory and uncertainty instead of a
binary ``detect_barren_plateau`` claim.
"""

import numpy as np
from typing import Dict, Sequence

_BOOTSTRAP_SEED = 0
_BOOTSTRAP_ITERATIONS = 2000


def _validate_matrix(value, name: str) -> np.ndarray:
    """Validate a 2-D float array and return it as float64."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D array of shape (rows, parameters), "
            f"got shape {arr.shape}"
        )
    if arr.shape[1] < 1:
        raise ValueError(f"{name} must have at least one parameter column")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


class GradientTracker:
    """Training diagnostic: per-parameter gradient variance over samples.

    ``update`` consumes a per-sample gradient matrix ``(B, P)`` and records
    ``V̄^x = mean_j Var_b[g_{b,j}]`` (mean over parameters of the variance over
    samples), the mean absolute gradient and the max absolute gradient, each as
    a per-step trajectory. ``get_statistics`` emits the metrics schema.
    """

    def __init__(self) -> None:
        self._steps: list = []
        self._vbar_x: list = []
        self._mean_abs_grad: list = []
        self._max_abs_grad: list = []
        self._n_samples: list = []

    def update(
        self, grad_matrix: Sequence, step: int = None, samples: int = None
    ) -> None:
        """Record a per-sample gradient matrix ``(B, P)``.

        Args:
            grad_matrix: Array of shape ``(B, P)`` — one gradient per training
                sample, one entry per trainable parameter.
            step: Global training-step index this diagnostic was logged at. If
                None, the running log index is used.
            samples: Number of samples ``B`` (inferred if not given).
        """
        g = _validate_matrix(grad_matrix, "grad_matrix")
        b, p = g.shape
        var_per_param = np.var(g, axis=0)  # (P,) variance over samples
        self._steps.append(int(step) if step is not None else len(self._steps))
        self._vbar_x.append(float(np.mean(var_per_param)))
        self._mean_abs_grad.append(float(np.mean(np.abs(g))))
        self._max_abs_grad.append(float(np.max(np.abs(g))))
        self._n_samples.append(int(samples) if samples is not None else int(b))

    @property
    def n_logged_steps(self) -> int:
        return len(self._steps)

    @property
    def is_empty(self) -> bool:
        return len(self._steps) == 0

    def state_dict(self) -> Dict:
        """Serializable snapshot of the accumulation (for checkpoints)."""
        return {
            key: list(getattr(self, key))
            for key in (
                '_steps', '_vbar_x', '_mean_abs_grad', '_max_abs_grad',
                '_n_samples',
            )
        }

    def restore_state(self, state: Dict) -> None:
        """Reload a snapshot produced by ``state_dict`` (no-op if absent)."""
        for key in (
            '_steps', '_vbar_x', '_mean_abs_grad', '_max_abs_grad', '_n_samples',
        ):
            if key in state:
                setattr(self, key, list(state[key]))

    def get_statistics(self) -> Dict:
        """Emit the training diagnostic (metrics schema).

        Returns:
            dict with ``n_logged_steps``, ``mean_param_grad_variance``,
            ``std_param_grad_variance``, ``mean_abs_grad``, ``max_abs_grad`` and
            the ``trajectory`` (step-index vs ``V̄^x``).
        """
        if self.is_empty:
            return {
                'n_logged_steps': 0,
                'mean_param_grad_variance': 0.0,
                'std_param_grad_variance': 0.0,
                'mean_abs_grad': 0.0,
                'max_abs_grad': 0.0,
                'trajectory': {
                    'step': [],
                    'mean_param_grad_variance': [],
                },
            }

        vbar_x = np.asarray(self._vbar_x, dtype=np.float64)
        return {
            'n_logged_steps': len(self._steps),
            'mean_param_grad_variance': float(np.mean(vbar_x)),
            'std_param_grad_variance': float(np.std(vbar_x)),
            'mean_abs_grad': float(np.mean(self._mean_abs_grad)),
            'max_abs_grad': float(np.max(self._max_abs_grad)),
            'trajectory': {
                'step': list(self._steps),
                'mean_param_grad_variance': list(self._vbar_x),
            },
        }


def landscape_variance(
    grad_instances: Sequence,
    n_bootstrap: int = _BOOTSTRAP_ITERATIONS,
    seed: int = _BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
) -> Dict:
    """Landscape statistic ``V̄`` over random parameter draws.

    Given ``R`` random instances of the cost-function gradient, each of length
    ``P`` (one entry per trainable parameter), computes:

    * ``V_j = Var_i[g_{i,j}]`` — variance over instances, per parameter;
    * ``V̄ = mean_j V_j`` — the landscape statistic;
    * Monte-Carlo uncertainty on ``V̄``: a nonparametric bootstrap
      that **resamples the ``R`` instances** (the Monte-Carlo draws), recomputing
      ``V̄`` each resample, reported as a bootstrap SE and a percentile CI.

    Args:
        grad_instances: Array of shape ``(R, P)``.
        n_bootstrap: Number of bootstrap resamples (over instances).
        seed: RNG seed for reproducibility of the bootstrap.
        confidence_level: Coverage of the percentile CI (e.g. 0.95).

    Returns:
        dict with ``Vbar``, ``variance_per_parameter``, ``se``, ``ci``,
        ``n_instances``, ``n_parameters``, ``n_bootstrap``.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")

    g = _validate_matrix(grad_instances, "grad_instances")
    r, p = g.shape
    v_j = np.var(g, axis=0)  # per-parameter variance over instances
    vbar = float(np.mean(v_j))

    rng = np.random.RandomState(seed)
    alpha = 1.0 - confidence_level
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.randint(0, r, size=r)
        boot[i] = np.mean(np.var(g[idx], axis=0))
    se = float(np.std(boot))
    lo, hi = np.percentile(boot, [100.0 * alpha / 2, 100.0 * (1 - alpha / 2)])

    return {
        'Vbar': vbar,
        'variance_per_parameter': v_j.tolist(),
        'se': se,
        'ci': [float(lo), float(hi)],
        'n_instances': int(r),
        'n_parameters': int(p),
        'n_bootstrap': int(n_bootstrap),
    }


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Classification accuracy as a fraction in ``[0, 1]``."""
    pred_labels = (predictions > 0.5).astype(int)
    if len(pred_labels) == 0:
        raise ValueError("cannot compute accuracy on an empty batch")
    return float(np.mean(pred_labels == labels))


def compare_approaches(results_dict: Dict[str, Dict]) -> Dict:
    """Summarise per-approach results using the new metrics schema.

    Args:
        results_dict: Mapping of approach name -> trainer result dict.

    Returns:
        A per-approach summary plus ``summary`` with the best test accuracy.
    """
    comparison = {}
    for name, results in results_dict.items():
        diag = results.get('training_diagnostic', {}) or {}
        comparison[name] = {
            'test_acc': results.get('test_acc'),
            'test_loss': results.get('test_loss'),
            'training_time_seconds': results.get('training_time_seconds'),
            'total_updates': results.get('total_updates'),
            'n_parameters': diag.get('n_parameters'),
            'mean_param_grad_variance': diag.get('mean_param_grad_variance'),
        }

    ranked = [
        (name, entry) for name, entry in comparison.items()
        if entry['test_acc'] is not None
    ]
    comparison['summary'] = {
        'best_test_acc': max(ranked, key=lambda x: x[1]['test_acc'])[0]
        if ranked else None,
    }
    return comparison