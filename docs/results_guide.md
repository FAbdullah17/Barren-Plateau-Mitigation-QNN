# Results Interpretation Guide

How to understand and analyze experiment results.

---

## Metrics Overview

Each experiment produces a `metrics.json` file with the following key metrics:

| Metric | Description | Good Value |
|--------|-------------|------------|
| `test_acc` | Final test accuracy | > 0.70 (70%) |
| `final_train_acc` | Final training accuracy | > 0.70 |
| `training_time` | Time in seconds | Varies by depth |
| `barren_plateau_detected` | Gradient vanishing flag | `false` |
| `gradient_stats.mean_norm` | Average gradient magnitude | > 1e-4 |

---

## Understanding Accuracy

### Actual Accuracy by Configuration (Mean ± Std across 5 seeds)

| Approach | 4 Layers | 6 Layers | 8 Layers |
|----------|----------|----------|----------|
| Baseline | 73.8% ± 2.2% | 73.9% ± 2.1% | **52.7% ± 1.1%** ⚠️ |
| Layerwise | 74.0% ± 3.0% | 74.2% ± 2.5% | 73.9% ± 2.6% |
| Local Cost | 75.3% ± 3.2% | 75.6% ± 3.1% | 75.4% ± 2.5% |

### Key Observations

1. **Baseline degrades catastrophically at 8 layers** — Accuracy drops to ~53% (near random chance), demonstrating the barren plateau problem
2. **Layerwise maintains performance** — Consistent ~74% across all depths, validating the mitigation strategy
3. **Local Cost provides the best and most stable results** — Consistently ~75% with no degradation at depth

---

## Understanding Gradients

### Gradient Norm Interpretation

| Mean Gradient Norm | Interpretation |
|--------------------|----------------|
| > 0.1 | Excellent — strong learning signal |
| 0.01 - 0.1 | Good — healthy gradients |
| 0.001 - 0.01 | Acceptable — may train slowly |
| < 0.001 | Warning — potential barren plateau |
| < 1e-6 | Barren Plateau Detected |

### Barren Plateau Detection

A barren plateau is flagged when:
- Mean gradient norm < 1e-6
- Training loss remains flat
- Accuracy near random (50%)

> **Note:** The 8-layer baseline results show barren plateau *behavior* (accuracy stuck at ~53%, flat loss) even though gradient norms remain above the 1e-6 detection threshold. This indicates gradient degradation severe enough to prevent learning, but not extreme enough to trigger the automated flag.

---

## Comparing Approaches

### What to Look For

1. **At 4 layers:** All approaches perform similarly (~74-75%)
2. **At 6 layers:** All approaches still perform similarly (~74-76%)
3. **At 8 layers:**
   - Baseline: Collapses to ~53% (barren plateau)
   - Layerwise: Maintains ~74% accuracy
   - Local Cost: Maintains ~75% accuracy

### Success Criteria

An approach is considered effective if:
- Test accuracy > 70%
- No significant accuracy drop with increasing depth
- Gradient norm > 0.01
- Training converges (loss decreases)

---

## Training Curves

The `training_history.png` plot shows:

1. **Loss curves** (top left)
   - Training loss (blue)
   - Validation loss (orange)
   - Should decrease over epochs

2. **Accuracy curves** (top right)
   - Training accuracy (blue)
   - Validation accuracy (orange)
   - Should increase over epochs

3. **Gradient norms** (bottom left, log scale)
   - Shows gradient magnitude over training
   - Should remain stable, not decay to zero

4. **Gradient variance** (bottom right, log scale)
   - Shows gradient stability over training

### Healthy vs Unhealthy Training

| Aspect | Healthy | Unhealthy |
|--------|---------|-----------|
| Loss | Decreases steadily | Flat or erratic |
| Accuracy | Increases to 70%+ | Stuck at ~50% |
| Gradients | Stable > 0.01 | Decaying toward 0 |

---

## Seed Variance

When running multiple seeds, expect:

- **Accuracy variance:** < 5% standard deviation
- **Gradient variance:** Within same order of magnitude
- **Training time:** Similar (±20%)

Use `analyze_seed_variance.py` to check:
```bash
python scripts/analyze_seed_variance.py results/baseline/depth_4/
```

---

## Result Validation

### Required Files
Each experiment should produce:
- `metrics.json` — All metrics
- `training_history.png` — Training curves

### Validation Command
```bash
python scripts/validate_results.py results/ -v
```

### Check Output Format
```bash
python scripts/check_output_format.py results/
```

---

## Expected Training Times

Based on production runs:

| Depth | Expected Time | Expected Accuracy |
|-------|---------------|-------------------|
| 4 layers | 10-30 min | 73-80% |
| 6 layers | 27-35 min | 70-80% |
| 8 layers | 38-60 min | 50-78%* |

*8-layer baseline expected to show barren plateau (~53%)

---

## Key Findings to Document

When analyzing results, focus on:

1. **Does baseline degrade at 8 layers?** (Should be YES — drops to ~53%)
2. **Does layerwise maintain performance?** (Should be YES — stays at ~74%)
3. **Does local cost maintain performance?** (Should be YES — stays at ~75%)
4. **Is the degradation statistically significant?** (Yes — 8L baseline is >20 percentage points below mitigation strategies)
5. **Is reproducibility confirmed?** (Variance < 5% across seeds)

These are the core findings for the research.

---

**Last Updated:** February 2026
