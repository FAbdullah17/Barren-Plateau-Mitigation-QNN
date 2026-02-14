# Results Interpretation Guide

How to understand and analyze experiment results.

---

## Metrics Overview

Each experiment produces a `metrics.json` file with the following key metrics:

| Metric | Description | Good Value |
|--------|-------------|------------|
| `test_acc` | Final test accuracy | > 0.80 (80%) |
| `train_acc` | Final training accuracy | > 0.75 |
| `training_time` | Time in seconds | Varies by depth |
| `barren_plateau_detected` | Gradient vanishing flag | `false` |
| `gradient_stats.mean_norm` | Average gradient magnitude | > 1e-4 |

---

## Understanding Accuracy

### Expected Accuracy by Configuration

| Approach | 4 Layers | 6 Layers | 8 Layers |
|----------|----------|----------|----------|
| Baseline | 75-80% | 70-80% | <80% ⚠️ |
| Layerwise | 78-82% | 78-82% | 78-82% ✅ |
| Local Cost | 78-82% | 76-80% | 75-80% |

### Key Observations

1. **Baseline degrades with depth** - This demonstrates the barren plateau problem
2. **Layerwise maintains performance** - Validates mitigation strategy
3. **Local Cost provides moderate improvement** - Alternative approach

---

## Understanding Gradients

### Gradient Norm Interpretation

| Mean Gradient Norm | Interpretation |
|--------------------|----------------|
| > 0.1 | Excellent - strong learning signal |
| 0.01 - 0.1 | Good - healthy gradients |
| 0.001 - 0.01 | Acceptable - may train slowly |
| < 0.001 | Warning - potential barren plateau |
| < 1e-6 | Barren Plateau Detected |

### Barren Plateau Detection

A barren plateau is flagged when:
- Mean gradient norm < 1e-6
- Training loss remains flat
- Accuracy near random (50%)

---

## Comparing Approaches

### What to Look For

1. **At 4 layers:** All approaches should perform similarly
2. **At 6 layers:** Baseline should show slight degradation
3. **At 8 layers:** 
   - Baseline: Should struggle (<80% or training failure)
   - Layerwise: Should maintain >75% accuracy
   - Local Cost: Should be between baseline and layerwise

### Success Criteria

An approach is considered successful if:
- Test accuracy > 75%
- No barren plateau detected
- Gradient norm > 1e-4
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

3. **Gradient norms** (bottom)
   - Shows gradient magnitude over training
   - Should remain stable, not decay to zero

### Healthy vs Unhealthy Training

| Aspect | Healthy | Unhealthy |
|--------|---------|-----------|
| Loss | Decreases steadily | Flat or erratic |
| Accuracy | Increases to 70%+ | Stuck at ~50% |
| Gradients | Stable > 0.01 | Decaying to 0 |

---

## Seed Variance

When running multiple seeds, expect:

- **Accuracy variance:** < 5% standard deviation
- **Gradient variance:** Within same order of magnitude
- **Training time:** Similar (±10%)

Use `analyze_seed_variance.py` to check:
```bash
python scripts/analyze_seed_variance.py results/baseline/depth_4/
```

---

## Result Validation

### Required Files
Each experiment should produce:
- `metrics.json` - All metrics
- `training_history.png` - Training curves

### Validation Command
```bash
python scripts/validate_results.py results/ -v
```

### Check Output Format
```bash
python scripts/check_output_format.py results/
```

---

## Expected Timeline

Based on validation runs:

| Depth | Expected Time | Expected Accuracy |
|-------|---------------|-------------------|
| 4 layers | 10-20 min | 76-80% |
| 6 layers | 15-30 min | 74-80% |
| 8 layers | 20-45 min | 70-80%* |

*8-layer baseline expected to perform poorly

---

## Key Findings to Document

When analyzing results, focus on:

1. **Does baseline degrade at 8 layers?** (Should be YES)
2. **Does layerwise maintain performance?** (Should be YES)
3. **Is gradient decay visible in baseline?** (Should be YES)
4. **Is reproducibility confirmed?** (Variance < 5%)

These are the core findings for the research paper.

---

**Last Updated:** January 2026
