# Metrics Schema Documentation

This document describes the structure of `metrics.json` files produced by all experiment runners.

## Required Fields

All experiment outputs contain the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `config` | object | Complete experiment configuration |
| `seed` | int | Random seed used for reproducibility |
| `final_train_loss` | float | Training loss at last epoch |
| `final_train_acc` | float | Training accuracy at last epoch (0-1) |
| `final_val_loss` | float | Validation loss at last epoch |
| `final_val_acc` | float | Validation accuracy at last epoch (0-1) |
| `test_loss` | float | Final test loss |
| `test_acc` | float | Final test accuracy (0-1) |
| `training_time` | float | Total training time in seconds |
| `gradient_stats` | object | Gradient statistics (see below) |
| `barren_plateau_detected` | bool | Whether barren plateau was detected |
| `history` | object | Training history (see below) |

## Gradient Statistics

The `gradient_stats` object contains:

| Field | Type | Description |
|-------|------|-------------|
| `mean_norm` | float | Mean gradient norm across training |
| `variance` | float | Variance of gradient norms |
| `min_norm` | float | Minimum gradient norm observed |
| `max_norm` | float | Maximum gradient norm observed |

## Training History

The `history` object contains arrays with one value per epoch:

| Field | Type | Description |
|-------|------|-------------|
| `train_loss` | float[] | Training loss per epoch |
| `train_acc` | float[] | Training accuracy per epoch |
| `val_loss` | float[] | Validation loss per epoch |
| `val_acc` | float[] | Validation accuracy per epoch |
| `gradient_norms` | float[] | Mean gradient norm per epoch |
| `gradient_variance` | float[] | Gradient variance per epoch |

## Example

```json
{
  "config": {
    "experiment": {"name": "baseline", "approach": "baseline"},
    "model": {"n_qubits": 4, "n_layers": 4},
    ...
  },
  "seed": 42,
  "final_train_loss": 0.5516,
  "final_train_acc": 0.728,
  "final_val_loss": 0.5176,
  "final_val_acc": 0.765,
  "test_loss": 0.5176,
  "test_acc": 0.765,
  "training_time": 637.88,
  "gradient_stats": {
    "mean_norm": 0.279,
    "variance": 0.0012,
    "min_norm": 0.168,
    "max_norm": 0.330
  },
  "barren_plateau_detected": false,
  "history": {
    "train_loss": [0.622, 0.567, ...],
    "train_acc": [0.634, 0.720, ...],
    ...
  }
}
```

## Barren Plateau Detection

A barren plateau is detected when:
- Mean gradient norm < 1e-6
- Gradient variance is very small

This indicates the optimization landscape is flat and training is ineffective.
