# Quick Start Guide

Get up and running with Hybrid-QNN experiments quickly.

---

## Prerequisites

Ensure you have the environment set up:
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Running Single Experiments

### Baseline Approach
```bash
python experiments/run_baseline.py configs/baseline_4layer.yaml --seed 42
```

### Layerwise Approach
```bash
python experiments/run_layerwise.py configs/layerwise_4layer.yaml --seed 42
```

### Local Cost Approach
```bash
python experiments/run_local_cost.py configs/local_cost_4layer.yaml --seed 42
```

---

## Running Batch Experiments

### Run all seeds for one approach
```bash
python scripts/run_batch.py baseline configs/baseline_4layer.yaml
```

### Run all experiments for a specific depth
```bash
# 4-layer (15 total: 3 approaches × 5 seeds)
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
python scripts/validate_results.py results/ -v
```

### Check output format consistency
```bash
python scripts/check_output_format.py results/
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
│       └── seed_{42,123,456,789,101112}/
│           ├── metrics.json
│           └── training_history.png
├── layerwise/
│   └── depth_{4,6,8}/
│       └── seed_{...}/
│           ├── metrics.json
│           └── training_history.png
└── local_cost/
    └── depth_{4,6,8}/
        └── seed_{...}/
            ├── metrics.json
            └── training_history.png
```

### Key metrics to check
- `test_acc` — Final test accuracy (0-1)
- `training_time` — Time in seconds
- `barren_plateau_detected` — True if gradients vanished below 1e-6
- `gradient_stats.mean_norm` — Average gradient magnitude

---

## Time Estimates

| Experiment Set | Estimated Time |
|----------------|----------------|
| Single 4-layer run | 10-30 min |
| Single 6-layer run | 27-35 min |
| Single 8-layer run | 38-60 min |
| All 4-layer (15 runs) | 5-8 hours |
| All 6-layer (15 runs) | 8-12 hours |
| All 8-layer (15 runs) | 12-20 hours |
| **ALL 45 runs** | **25-40 hours** |

---

## Common Commands

```bash
# Run a quick smoke test
python experiments/run_baseline.py configs/baseline_test.yaml --seed 42

# Run full experiment
python experiments/run_baseline.py configs/baseline_4layer.yaml --seed 42

# Validate results
python scripts/validate_results.py results/

# Run data consistency tests
python tests/test_data_consistency.py
```

---

## Troubleshooting

See [troubleshooting.md](troubleshooting.md) for common issues and solutions.

---

**Last Updated:** February 2026
