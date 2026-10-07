# Metrics Schema Documentation

This document describes the structure of `metrics.json` files produced by all
experiment runners (`experiments/run_baseline.py`, `run_layerwise.py`,
`run_local_cost.py`).

## Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `config` | object | Resolved run configuration (approach, model, training hyperparameters, seeds used) |
| `data_seed` | int | Seed used for data subsampling (derived from `seeds.base_seed`) |
| `init_seed` | int | Seed used for parameter initialization |
| `training_seed` | int | Seed used for dataset shuffling |
| `seed_index` | int | Seed-triple index (`derive_seed_triple(base_seed, index)`) |
| `test_loss` | float | Final test (validation) loss |
| `test_acc` | float | Final test (validation) accuracy (0-1) |
| `training_time_seconds` | float | Total training time in seconds |
| `total_updates` | int | Total number of gradient updates consumed |
| `layerwise_budget_split` | object \| null | Layerwise update budget `{per_stage, finetune}`; `null` for non-layerwise runs |
| `n_parameters` | int | **Runtime-derived** parameter count of the final model |
| `training_diagnostic` | object | Gradient statistics (see below) |
| `history` | object | Step-based training history (see below) |
| `pca_info` | object | PCA fit statistics from `prepare_features` |

> `n_parameters` is always read from the instantiated model at runtime — never
> from config or manual calculation.

## Training Diagnostic

The `training_diagnostic` object contains:

| Field | Type | Description |
|-------|------|-------------|
| `n_parameters` | int | Runtime-derived parameter count |
| `mean_param_grad_variance` | float | `V̄ˣ` = mean over parameters of the variance-over-samples of the per-parameter gradient |
| `std_param_grad_variance` | float | Std over parameters of the variance-over-samples |
| `mean_abs_grad` | float | Mean |∂ℓ/∂θ| over samples and parameters |
| `max_abs_grad` | float | Maximum |∂ℓ/∂θ| |
| `trajectory` | object | Logged steps (see below) |

The `trajectory` object:

| Field | Type | Description |
|-------|------|-------------|
| `step` | int[] | Gradient steps at which the diagnostic was logged |
| `mean_param_grad_variance` | float[] | `V̄ˣ` per logged step |

## Training History

The `history` object contains arrays with one value per gradient step
(training metrics) or per logged step (validation metrics):

| Field | Type | Description |
|-------|------|-------------|
| `step` | int[] | Gradient step indices (0-based) |
| `train_loss` | float[] | Training loss per step |
| `train_acc` | float[] | Training accuracy per step (0-1) |
| `val_step` | int[] | Steps at which validation was evaluated |
| `val_loss` | float[] | Validation loss per `val_step` |
| `val_acc` | float[] | Validation accuracy per `val_step` (0-1) |

## `pca_info`

| Field | Type | Description |
|-------|------|-------------|
| `n_components` | int | Number of PCA components (= qubits) |
| `image_size` | int[] | Downsampled image size `[height, width]` |
| `explained_variance_ratio` | float[] | Per-component explained variance |
| `cumulative_explained_variance` | float | Total explained variance of the retained components |
| `train_min` | float[] | Per-feature min used for normalization (train only) |
| `train_span` | float[] | Per-feature span used for normalization (train only) |

## Example

```json
{
  "config": {
    "approach": "baseline",
    "n_qubits": 6,
    "n_layers": 8,
    "cost": "global",
    "learning_rate": 0.01,
    "batch_size": 20,
    "total_updates": 2500,
    "log_frequency": 10,
    "init_seed": 43,
    "training_seed": 44,
    "track_gradients": true
  },
  "data_seed": 42,
  "init_seed": 43,
  "training_seed": 44,
  "seed_index": 0,
  "test_loss": 0.4812,
  "test_acc": 0.785,
  "training_time_seconds": 1523.1,
  "total_updates": 2500,
  "layerwise_budget_split": null,
  "n_parameters": 96,
  "training_diagnostic": {
    "n_parameters": 96,
    "mean_param_grad_variance": 0.00031,
    "std_param_grad_variance": 0.00012,
    "mean_abs_grad": 0.0191,
    "max_abs_grad": 0.0842,
    "trajectory": {
      "step": [9, 19, 29, "...", 2499],
      "mean_param_grad_variance": [0.0004, 0.00035, "...", 0.0003]
    }
  },
  "history": {
    "step": [0, 1, 2, "...", 2499],
    "train_loss": [0.596, 0.565, "...", 0.489],
    "train_acc": [0.634, 0.720, "...", 0.791],
    "val_step": [9, 19, 29, "...", 2499],
    "val_loss": [0.570, 0.540, "...", 0.481],
    "val_acc": [0.680, 0.730, "...", 0.785]
  },
  "pca_info": {
    "n_components": 6,
    "image_size": [4, 4],
    "explained_variance_ratio": [0.32, 0.21, 0.15, 0.11, 0.09, 0.07],
    "cumulative_explained_variance": 0.95,
    "train_min": [...],
    "train_span": [...]
  }
}
```

## Notes

- **No binary barren-plateau claim** is emitted: `barren_plateau_detected` and
  the fixed `1e-6` threshold are removed. Gradient behavior is reported through
  the trajectories and the variance statistics above.
- `scripts/validate_results.py` and `scripts/check_output_format.py` enforce
  these required fields and files.