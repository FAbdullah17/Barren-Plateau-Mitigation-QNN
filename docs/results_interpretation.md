# Results Interpretation Guide

Detailed analysis guide with actual data from the 180 production runs.

## Understanding Experimental Outputs

### Metrics JSON Structure

Each experiment generates a `metrics.json` file (see [Metrics Schema](metrics_schema.md) for the authoritative spec):

```json
{
  "config": { "approach": "baseline", "n_qubits": 8, "n_layers": 4, "cost": "global",
              "learning_rate": 0.01, "batch_size": 20, "total_updates": 2500 },
  "data_seed": 42,
  "init_seed": 43,
  "training_seed": 44,
  "seed_index": 0,
  "test_loss": 0.48,
  "test_acc": 0.885,
  "training_time_seconds": 15230,
  "total_updates": 2500,
  "layerwise_budget_split": null,
  "n_parameters": 64,
  "training_diagnostic": {
    "mean_param_grad_variance": 0.00694,
    "trajectory": { "step": [...], "mean_param_grad_variance": [...] }
  },
  "history": {
    "step": [...], "train_loss": [...], "train_acc": [...],
    "val_step": [...], "val_loss": [...], "val_acc": [...]
  },
  "pca_info": { "n_components": 8, ... }
}
```

---

## Interpreting Key Metrics

### 1. Test Accuracy

**What it measures**: Classification performance on held-out test data.

**Observed ranges** (180 runs, 20 seed triples per condition):
- Every approach × depth combination averages **~86-87%**
- Per-condition range of the mean: 86.1% (baseline & local cost at 4L) to 87.3% (local cost at 8L)
- No run falls below 78.5%; no approach suffers depth degradation

**Actual results** (mean ± std, 20 seed triples):

| Approach | 4-Layer | 6-Layer | 8-Layer |
|----------|---------|---------|---------|
| Baseline | 86.1 ± 3.3% | 87.2 ± 1.8% | 87.2 ± 2.2% |
| Layerwise | 86.4 ± 3.0% | 86.7 ± 2.4% | 86.7 ± 2.9% |
| Local Cost | 86.1 ± 2.7% | 87.1 ± 2.2% | 87.3 ± 2.1% |

**Interpretation**: With 8 qubits, all strategies — including the baseline — train successfully at every depth. Accuracy is **not** the discriminating metric between approaches on this benchmark.

### 2. Gradient Variance (Primary Diagnostic)

**What it measures**: `mean_param_grad_variance` — the mean over parameters of the variance-over-samples of each parameter's gradient (V̄ˣ). Lower values indicate flatter/less variable gradients; the quantity is the standard barren-plateau diagnostic.

**Actual final values**:

| Config | Final Grad-Var | vs Baseline |
|--------|----------------|-------------|
| Baseline 4/6/8L | 0.00694 / 0.00658 / 0.00350 | 1.0× |
| Layerwise 4/6/8L | 0.00731 / 0.00628 / 0.00452 | 1.05 / 0.96 / 1.29× |
| Local Cost 4/6/8L | **0.00319 / 0.00279 / 0.00240** | **0.46 / 0.42 / 0.69×** |

**Interpretation**: The local cost function maintains 1.5-2.4× lower gradient variance than the global-cost baseline at every depth, at no accuracy cost. This is the statistically robust, reproducible effect of the study.

**Trajectory patterns**: baseline and local cost typically show gradient variance *rising* over training (16-19/20 and 14/20 runs respectively), while layerwise training shows it *falling* (0-1/20 runs). Layerwise is the only approach whose optimization path systematically settles the gradients.

### 3. Training Time

**Observed means** (8 qubits, on CPU):

| Config | Mean Time |
|--------|-----------|
| Baseline 4/6/8L | 4.2 / 1.7 / 1.7 h |
| Layerwise 4/6/8L | 1.1 / 0.7 / 1.8 h |
| Local Cost 4/6/8L | 3.5 / 4.1 / 6.6 h |

Local cost is the most expensive at depth; layerwise is the cheapest for shallow circuits.

---

## Statistical Significance

### Performed Analysis

Welch t-tests on the 20-seed-triple results for each approach-depth combination:

| Comparison | 4L p | 6L p | 8L p | Verdict |
|------------|------|------|------|---------|
| Baseline vs Local Cost (acc) | 0.959 | 0.879 | 0.801 | Not significant |
| Layerwise vs Local Cost (acc) | 0.748 | 0.551 | 0.434 | Not significant |
| Baseline vs Layerwise (acc) | 0.808 | 0.437 | 0.571 | Not significant |
| Baseline vs Local Cost (grad-var) | 7.8e-7 | 3.5e-9 | 6.8e-3 | **Highly significant** |
| Layerwise vs Local Cost (grad-var) | 1.2e-8 | 7.3e-9 | 1.8e-9 | **Highly significant** |

```python
from scipy import stats

baseline_8L_gv = [<20 final grad-var values>]
local_8L_gv = [<20 final grad-var values>]

t_stat, p_value = stats.ttest_ind(baseline_8L_gv, local_8L_gv, equal_var=False)
print(f"t = {t_stat:.3f}, p = {p_value:.3e}")
```

### Effect Size

- The accuracy effect between approaches is negligible (no separation).
- The gradient-variance effect for local cost vs baseline is **large** across all depths (Cohen's d = 1.9 / 2.7 / 0.9 for 4/6/8L, all above the 0.8 "large" threshold; p ≤ 0.007 with n = 20).

---

## Success Rate Analysis

**Definition**: Percentage of runs achieving ≥70% accuracy.

**Actual results (20 runs per cell)**:

```
Depth 4:
  Baseline:    100% (20/20 ≥ 70%)
  Layerwise:   100% (20/20 ≥ 70%)
  Local Cost:  100% (20/20 ≥ 70%)

Depth 6:
  Baseline:    100% (20/20 ≥ 70%)
  Layerwise:   100% (20/20 ≥ 70%)
  Local Cost:  100% (20/20 ≥ 70%)

Depth 8:
  Baseline:    100% (20/20 ≥ 70%)
  Layerwise:   100% (20/20 ≥ 70%)
  Local Cost:  100% (20/20 ≥ 70%)
```

At a stricter ≥85% threshold (4/6/8L): Baseline 75/90/90%, Layerwise 60/75/85%, Local Cost 60/85/90%.

**Key insight**: Every strategy succeeds on this benchmark at every depth; success-rate analysis does not separate the approaches.

---

## Depth Impact Analysis

### Observed Patterns

All three approaches are flat across depth:

```
Accuracy (%)
 90|●────────●────────●   all approaches ~86-87%
 85|
 80|
 75|
 70|
   └──────────────────
     4    6    8  Depth
```

There is **no barren-plateau collapse** at this scale: baseline, layerwise, and local cost all converge at 8 layers. The mitigation signal appears in gradient variance instead:

```
Final grad-var
 0.007|●(base 4L)  ●(base 6L)
 0.005|             ●(layer 8L)
 0.003|●(local 4L)  ●(local 6L)
      |                       ●(local 8L)
      └──────────────────
        4    6    8  Depth
```

Local cost is consistently lowest; the baseline is consistently highest.

---

## Visualization Interpretation

### 1. Training Loss Curves

**Healthy training**:
- Smooth decrease over the 2500 updates
- Convergence is reliable across all 20 seeds per condition

### 2. Gradient Trajectory Plots

- **Baseline / Local Cost**: gradient variance rises over training (16-19/20 and 14/20 runs) — gradients strengthen as the model learns
- **Layerwise**: gradient variance falls over training (0-1/20 runs) — the incremental schedule settles the gradients

This dichotomy is the clearest visual signature for distinguishing layerwise training from the other two strategies.

### 3. Comparison Bar Charts

**Error bars interpretation**:
- Accuracy error bars (1.8-3.3%) heavily overlap between approaches → no significant difference
- Gradient-variance error bars (local cost: 0.0008-0.0015, baseline: 0.0017-0.0023) overlap only slightly → significant reduction

---

## Common Result Patterns

### Pattern 1: No Accuracy Separation at Depth
```
Approach     | 4L Acc | 8L Acc | Delta
-------------|--------|--------|------
Baseline     | 86.1%  | 87.2%  | +1.1 pp  (no collapse)
Layerwise    | 86.4%  | 86.7%  | +0.3 pp
Local Cost   | 86.1%  | 87.3%  | +1.2 pp
```
**Conclusion**: At 8 qubits the benchmark is trainable by all strategies; accuracy does not separate approaches.

### Pattern 2: Local Cost Lowers Gradient Variance
```
Depth | Baseline | Local Cost | Reduction | p
------|----------|------------|-----------|-----
4L    | 0.00694  | 0.00319    | 2.2×      | 7.8e-7
6L    | 0.00658  | 0.00279    | 2.4×      | 3.5e-9
8L    | 0.00350  | 0.00240    | 1.5×      | 6.8e-3
```
**Conclusion**: Local cost functions deliver a statistically significant, reproducible reduction in gradient variance at all depths without hurting accuracy.

### Pattern 3: Layerwise Settles Gradients
Layerwise is the only approach whose gradient-variance trajectory *decreases* during training (1/20 runs rising, vs 16-19/20 for baseline), indicating a distinct optimization signature.

---

## Troubleshooting Results

### Problem: Low Accuracy Across All Approaches

**Possible causes**:
1. Data quality / PCA issues
2. Hyperparameters need tuning
3. Bug in implementation (e.g., label/gradient handling)

**Diagnosis steps**:
```python
# Check data
assert X_train.shape[1] == 8          # PCA-reduced features
assert set(y_train) == {-1, 1}

# Check initial gradient health
diag = metrics['training_diagnostic']
assert diag['mean_param_grad_variance'] > 1e-4

# Check loss decrease
h = metrics['history']
assert h['train_loss'][-1] < h['train_loss'][0]
```

### Problem: A Run Fails to Train (Loss Flat, ~50% Acc)

**Diagnosis**: A genuinely failed run should be inspected for:
- A seed triple that collides (verify `seed_index → (data, init, training)_seed` mapping)
- PCA fit statistics (`pca_info`) — e.g., a degenerate component count

**Solutions**:
1. Re-run the seed (the runner skips indices that are already complete; delete the seed's `metrics.json` to force a fresh run)
2. Reduce depth or increase update budget
3. Switch to local cost for gradient-variance safety margin

### Problem: High Variance Across Seeds

**Indicators**:
- Std dev > 5% of mean (observed max is 3.3%)
- Success rate < 90% at the ≥70% threshold

**Solutions**:
1. More seeds (the 20-triple ladder; extend `seeds.seed_triples` in the config)
2. Adjust learning rate (0.001-0.1)
3. Increase `total_updates`

---

**Last Updated:** October 2026