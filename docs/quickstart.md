# Quick Start Guide

Get up and running with Hybrid-QNN experiments quickly.

---

## Prerequisites

Ensure you have the environment set up:
```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Running Single Experiments

Each run consumes one seed triple. `--seed-index` selects the seed index (0-19); the
data/init/training seeds are derived as `42 + 3×index`, `+ 1`, `+ 2`.

### Baseline Approach
```bash
python experiments/run_baseline.py configs/baseline_4layer.yaml --seed-index 0
```

### Layerwise Approach
```bash
python experiments/run_layerwise.py configs/layerwise_4layer.yaml --seed-index 0
```

### Local Cost Approach
```bash
python experiments/run_local_cost.py configs/local_cost_4layer.yaml --seed-index 0
```

---

## Running Batch Experiments

### Run all 20 seeds for one approach
```bash
python scripts/run_batch.py baseline configs/baseline_4layer.yaml
```

### Run all experiments for a specific depth
```bash
# 4-layer (60 total: 3 approaches × 20 seeds)
python scripts/run_4layer_experiments.py

# 6-layer
python scripts/run_6layer_experiments.py

# 8-layer
python scripts/run_8layer_experiments.py
```

### Dry run (see commands without executing)
```bash
python scripts/run_4layer_experiments.py --dry-run
```

---

## Available Configurations

| Config File | Layers | Approach |
|-------------|--------|----------|
| `configs/baseline_4layer.yaml` | 4 | Baseline |
| `configs/baseline_6layer.yaml` | 6 | Baseline |
| `configs/baseline_8layer.yaml` | 8 | Baseline |
| `configs/layerwise_4layer.yaml` | 4 | Layerwise |
| `configs/layerwise_6layer.yaml` | 6 | Layerwise |
| `configs/layerwise_8layer.yaml` | 8 | Layerwise |
| `configs/local_cost_4layer.yaml` | 4 | Local Cost |
| `configs/local_cost_6layer.yaml` | 6 | Local Cost |
| `configs/local_cost_8layer.yaml` | 8 | Local Cost |

---

## Validating Results

### Validate all results
```bash
python3 scripts/validate_results.py results/baseline -v
python3 scripts/validate_results.py results/layerwise -v
python3 scripts/validate_results.py results/local_cost -v
```

### Check output format consistency
```bash
python3 scripts/check_output_format.py results
```

### Analyze seed variance
```bash
python scripts/analyze_seed_variance.py results/baseline/depth_4/
```

---

## Viewing Results

### Results location
```
results/
├── baseline/
│   └── depth_{4,6,8}/
│       └── seed_{0..19}/
│           ├── metrics.json
│           └── training_history.png
├── layerwise/
│   └── depth_{4,6,8}/
│       └── seed_{0..19}/
│           ├── metrics.json
│           └── training_history.png
└── local_cost/
    └── depth_{4,6,8}/
        └── seed_{0..19}/
            ├── metrics.json
            └── training_history.png
```

### Key metrics to check
- `test_acc` — Final test accuracy (0-1)
- `training_diagnostic.mean_param_grad_variance` — Final gradient variance (barren-plateau diagnostic)
- `training_diagnostic.trajectory` — Gradient-variance over training
- `training_time_seconds` — Time in seconds

Full schema: [Metrics Schema](metrics_schema.md).

---

## Time Estimates

Observed means from the 180 production runs:

| Experiment Set | Observed Mean Time |
|----------------|--------------------|
| Single 4-layer run | 1.1-4.2 h |
| Single 6-layer run | 0.7-4.1 h |
| Single 8-layer run | 1.7-6.6 h |
| **All 180 runs** | **~506 CPU-hours (~21 CPU-days)** |

---

## Common Commands

```bash
# Run a quick smoke test
python experiments/run_baseline.py configs/baseline_test.yaml --seed-index 0

# Run full experiment
python experiments/run_baseline.py configs/baseline_4layer.yaml --seed-index 7

# Validate results
python scripts/validate_results.py results/

# Run data consistency tests
python tests/test_data_consistency.py
```

---

## Troubleshooting

See [troubleshooting.md](troubleshooting.md) for common issues and solutions.

---

**Last Updated:** October 2026
