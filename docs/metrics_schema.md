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
| `std_norm` | float | Standard deviation of gradient norms |
| `variance` | float | Variance of gradient norms |
| `min_norm` | float | Minimum gradient norm observed |
| `max_norm` | float | Maximum gradient norm observed |
| `median_norm` | float | Median gradient norm observed |
| `total_updates` | float | Total number of gradient updates |

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
    "experiment": {"name": "baseline_4layer", "approach": "baseline"},
    "model": {"n_qubits": 4, "n_layers": 4},
    "training": {"optimizer": "adam", "learning_rate": 0.01, "batch_size": 20, "epochs": 50},
    "data": {"dataset": "mnist", "digit1": 3, "digit2": 6, "train_size": 1000, "test_size": 200},
    ...
  },
  "seed": 42,
  "final_train_loss": 0.5507,
  "final_train_acc": 0.730,
  "final_val_loss": 0.5176,
  "final_val_acc": 0.760,
  "test_loss": 0.5176,
  "test_acc": 0.760,
  "training_time": 1583.03,
  "gradient_stats": {
    "mean_norm": 0.2843,
    "std_norm": 0.1802,
    "variance": 0.0325,
    "min_norm": 0.0122,
    "max_norm": 1.1493,
    "median_norm": 0.2500,
    "total_updates": 2500.0
  },
  "barren_plateau_detected": false,
  "history": {
    "train_loss": [0.596, 0.565, ...],
    "train_acc": [0.634, 0.720, ...],
    "val_loss": [0.570, 0.540, ...],
    "val_acc": [0.680, 0.730, ...],
    "gradient_norms": [0.45, 0.38, ...],
    "gradient_variance": [0.08, 0.05, ...]
  }
}
```

## Barren Plateau Detection

A barren plateau is detected when:
- Mean gradient norm < 1e-6 (configurable via `barren_plateau_threshold` in config)
- This indicates the optimization landscape is flat and training is ineffective

> **Note:** Even when `barren_plateau_detected` is `false`, training may still fail to converge at deeper circuit depths (e.g., 8 layers). The 1e-6 threshold detects severe gradient vanishing; moderate gradient degradation may still cause training stagnation without triggering this flag.
