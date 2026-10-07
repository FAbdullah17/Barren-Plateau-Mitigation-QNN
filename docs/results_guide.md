# Results Interpretation Guide

How to understand and analyze experiment results.

---

## Metrics Overview

Each experiment produces a `metrics.json` file with the following key metrics:

| Metric | Description | Good Value |
|--------|-------------|------------|
| `test_acc` | Final test accuracy | > 0.70 (70%) |
| `test_loss` | Final test (validation) loss | decreasing over training |
| `training_time_seconds` | Time in seconds | Varies by depth/approach |
| `total_updates` | Gradient updates consumed | 2500 |
| `training_diagnostic.mean_param_grad_variance` | Final mean per-parameter gradient variance (V̄ˣ) | lower = healthier gradients |
| `training_diagnostic.trajectory` | Logged gradient-variance over training | see trajectory analysis |

> No binary `barren_plateau_detected` flag is emitted. Gradient behavior is
> reported through the variance statistics and trajectories (see the
> [Metrics Schema](metrics_schema.md)).

---

## Understanding Accuracy

Full accuracy tables and statistics are reported in the [README (§4 Results)](../README.md#4-results) and the [Results Report](results_report.md). In summary:

### Key Observations

1. **All approaches are statistically tied on accuracy** — pairwise Welch t-tests give p ≥ 0.43 at every depth
2. **No depth degradation** — every approach holds ~86-87% at 4, 6, and 8 layers (including the baseline)
3. **The variations are real but comparable** — run-to-run spread is 1.8-3.3% across 20 seed triples

Accuracy therefore does **not** separate mitigation strategies on this task; the differentiating signal is gradient variance (next section).

---

## Understanding Gradients

### Gradient Variance as the Key Diagnostic

The per-parameter gradient variance V̄ˣ (mean over parameters of the variance-over-samples of each gradient) is the primary barren-plateau diagnostic:

| Final Grad-Var | Interpretation |
|----------------|----------------|
| relative to baseline down to ~0.4× | Strong mitigation signal (local cost) |
| ~1.0× (same as baseline) | No change from end-to-end training |
| upward trajectory during training | Gradients diverge/strengthen as training proceeds |
| downward trajectory | Gradients settle as training proceeds (layerwise) |

### Observed Gradient Variance (Final)

The observed final gradient variances and their ratios to the baseline are tabulated in the [README (§4 Results)](../README.md#4-results) and the [Results Report](results_report.md). In summary: local cost consistently lowers gradient variance by 1.5-2.4× versus the global-cost baseline (Welch t-test p ≤ 0.007 at every depth), at no loss in accuracy.

### Trajectory Trends

| Approach | Dominant Grad-Var Trajectory | Rising runs |
|----------|------------------------------|-------------|
| Baseline | Rises during training | 16-19/20 |
| Layerwise | Falls during training | 0-1/20 |
| Local Cost | Rises mildly | 14/20 |

Layerwise training is unique in showing gradient variance that *decreases* as training progresses, reflecting its incremental optimization path.

---

## Comparing Approaches

### What to Look For

1. **At every depth:** all approaches achieve ~86-87% accuracy — differences are not significant (p ≥ 0.43)
2. **Gradient variance:** local cost is 1.5-2.4× lower than baseline at all depths (significant, p ≤ 0.007)
3. **Layerwise trajectory:** the only approach whose gradient variance falls during training

### Success Criteria

An approach is considered effective if:
- Test accuracy > 70%
- No significant accuracy drop with increasing depth
- Lower gradient variance than the global-cost baseline
- Training converges (loss decreases)

---

## Training Curves

The `training_history.png` plot shows two panels (over gradient steps):

1. **Loss curves** (left)
   - Training loss (blue)
   - Validation loss (orange)
   - Should decrease over updates

2. **Accuracy curves** (right)
   - Training accuracy (blue)
   - Validation accuracy (orange)
   - Should increase over updates

The gradient-variance trajectory is recorded in `metrics.json`
(`training_diagnostic.trajectory`) and can be plotted with
`src/evaluation/visualization.py::plot_gradient_trajectory`; compare the trend
(rising for baseline/local cost, falling for layerwise).

### Healthy vs Unhealthy Training

| Aspect | Healthy | Unhealthy |
|--------|---------|-----------|
| Loss | Decreases steadily | Flat or erratic |
| Accuracy | Increases to 70%+ | Stuck at ~50% |
| Gradient variance | Maintained or settling | Collapsing to ~0 or exploding |

---

## Seed Variance

When running multiple seeds, expect:

- **Accuracy variance:** < 5% standard deviation (observed 1.8-3.3%)
- **Gradient variance:** same order of magnitude across seeds
- **Training time:** similar (±20%)

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

Mean training times per configuration are reported in the [README (§4 Results)](../README.md#4-results) and the [Results Report](results_report.md). As a rule of thumb, local cost is the most expensive at depth, while layerwise training is the cheapest for shallow circuits.

---

## Key Findings to Document

When analyzing results, focus on:

1. **Do all approaches maintain accuracy at depth?** (YES — ~86-87% at 4/6/8L, no loss)
2. **Is there an accuracy difference between approaches?** (NO — p ≥ 0.43 at every depth)
3. **Does local cost lower gradient variance?** (YES — 1.5-2.4× lower than baseline, p ≤ 0.007)
4. **Is layerwise qualitatively different?** (YES — gradient variance *falls* during training)
5. **Is reproducibility confirmed?** (Yes — 1.8-3.3% std across 20 seed triples)

These are the core findings for the research.

---

**Last Updated:** October 2026