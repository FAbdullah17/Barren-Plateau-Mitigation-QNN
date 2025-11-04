# Results Interpretation Guide

## Understanding Experimental Outputs

### Training History Files

Each experiment generates a JSON file containing:

```json
{
  "train_loss": [0.45, 0.32, 0.21, ...],
  "test_loss": [0.48, 0.35, 0.24, ...],
  "train_accuracy": [65.2, 78.5, 88.3, ...],
  "test_accuracy": [63.8, 76.1, 86.2, ...],
  "gradient_norms": [0.152, 0.089, 0.034, ...],
  "gradient_variances": [0.023, 0.012, 0.004, ...],
  "has_barren_plateau": false,
  "final_test_accuracy": 92.5,
  "training_time": 245.3
}
```

### Comparison CSV Structure

The comprehensive comparison generates a CSV with columns:

| Column | Description | Expected Range |
|--------|-------------|----------------|
| `approach` | Training method | baseline, layerwise, local_cost |
| `depth` | Circuit layers | 4, 6, 8 |
| `seed` | Random seed | 42, 123, 456, 789, 101112 |
| `final_test_accuracy` | Test accuracy | 0-100% |
| `training_time` | Training duration | seconds |
| `has_barren_plateau` | BP detected | True/False |
| `mean_gradient_norm` | Average gradient | typically 1e-6 to 1e-1 |

---

## Interpreting Key Metrics

### 1. Test Accuracy

**What it measures**: Classification performance on unseen data

**Good values**:
- ≥90%: Excellent performance
- 80-90%: Good performance
- 70-80%: Moderate performance
- <70%: Poor performance (likely barren plateau)

**Comparison**:
```python
baseline_acc = 82.3 ± 5.1%
layerwise_acc = 91.2 ± 2.4%  # Better mean, lower variance
local_cost_acc = 89.7 ± 3.8%
```

**Interpretation**: Layerwise achieves highest accuracy with most consistency.

### 2. Gradient Norms

**What it measures**: Magnitude of parameter updates

**Healthy trajectory**:
- Starts: ~0.1 - 0.5
- Mid-training: ~0.01 - 0.1
- Converged: ~0.001 - 0.01

**Barren plateau signature**:
- Drops to <1e-6 within first 10 epochs
- Remains flat (no improvement)

**Visual patterns**:

```
Normal gradient decay:
Norm
0.5 |●
0.1 |  ●●
0.01|    ●●●
1e-3|       ●●●●
    └────────────→ Epoch

Barren plateau:
Norm
0.5 |●
1e-6|●●●●●●●●●●●●●
    └────────────→ Epoch
```

### 3. Gradient Variance

**What it measures**: Stability of gradient updates

**Interpretation**:
- High variance (>0.1): Unstable training, needs smaller learning rate
- Medium variance (0.001-0.1): Normal training
- Very low variance (<1e-6): Possible barren plateau

**Relationship to barren plateaus**:

$$\text{Var}[\nabla L] \propto \frac{1}{2^n} \quad \text{(barren plateau)}$$

### 4. Training Time

**Factors**:
- Circuit depth (deeper = slower)
- Number of epochs
- Batch size

**Expected times** (4 qubits, 50 epochs):
- Baseline (4 layers): ~2-3 minutes
- Baseline (8 layers): ~4-5 minutes
- Layerwise (4 layers): ~3-4 minutes (more epochs)
- Local cost (4 layers): ~2-3 minutes (similar to baseline)

**Trade-off analysis**:
```
Approach     | Time  | Accuracy | Efficiency
-------------|-------|----------|------------
Baseline     | 2.5m  | 82%      | 32.8%/min
Layerwise    | 3.5m  | 91%      | 26.0%/min  ← Best accuracy
Local Cost   | 2.3m  | 90%      | 39.1%/min  ← Most efficient
```

---

## Statistical Significance

### T-Test Results

```
Baseline vs Layerwise: p = 0.0023 **
Baseline vs Local Cost: p = 0.0156 *
Layerwise vs Local Cost: p = 0.3421 n.s.
```

**Significance levels**:
- `***`: p < 0.001 (highly significant)
- `**`: p < 0.01 (very significant)
- `*`: p < 0.05 (significant)
- `n.s.`: p ≥ 0.05 (not significant)

**Interpretation**: Layerwise and local cost both significantly outperform baseline, but are not significantly different from each other.

### Effect Size (Cohen's d)

```
Baseline vs Layerwise: d = 0.82 (large effect)
Baseline vs Local Cost: d = 0.71 (medium effect)
Layerwise vs Local Cost: d = 0.15 (small effect)
```

**Guidelines**:
- d < 0.2: Negligible
- 0.2 ≤ d < 0.5: Small
- 0.5 ≤ d < 0.8: Medium
- d ≥ 0.8: Large

---

## Success Rate Analysis

**Definition**: Percentage of runs achieving ≥90% accuracy

**Example results**:
```
Depth 4:
  Baseline:    60% (3/5 runs)
  Layerwise:   100% (5/5 runs) ✓
  Local Cost:  80% (4/5 runs)

Depth 6:
  Baseline:    20% (1/5 runs)
  Layerwise:   80% (4/5 runs)
  Local Cost:  80% (4/5 runs)

Depth 8:
  Baseline:    0% (0/5 runs)    ← Barren plateau
  Layerwise:   60% (3/5 runs)
  Local Cost:  60% (3/5 runs)
```

**Key insights**:
- Baseline degrades severely with depth
- Mitigation strategies maintain performance longer
- Even mitigation fails at extreme depth (8+ layers)

---

## Depth Impact Analysis

### Expected Patterns

**Baseline approach**:
```
Accuracy
100%|
 90%|●●●
 80%|   ●
 70%|     ●
 60%|       ●
    └─────────────→
    4  6  8  Depth
```
Sharp degradation with depth (barren plateau effect)

**Layerwise approach**:
```
Accuracy
100%|
 90%|●●●●●
 80%|     ●
 70%|
 60%|
    └─────────────→
    4  6  8  Depth
```
Graceful degradation, maintains performance longer

**Local cost approach**:
```
Accuracy
100%|
 90%|●●●●
 80%|    ●
 70%|
 60%|
    └─────────────→
    4  6  8  Depth
```
Similar to layerwise, slightly less robust

---

## Visualization Interpretation

### 1. Training Loss Curves

**Healthy training**:
- Smooth decrease
- Train and test losses track together
- Convergence within 30-50 epochs

**Problematic patterns**:
- **Barren plateau**: Loss flat from start, no decrease
- **Overfitting**: Train loss decreases, test loss increases
- **Oscillation**: Erratic jumps, needs lower learning rate

### 2. Gradient Trajectory Plots

**Normal behavior**:
```
log(Gradient)
-1 |●
-2 |  ●●
-3 |    ●●●
-4 |       ●●●●
-5 |          ●●●●●
   └────────────────→ Epoch
```
Exponential decay on log scale = linear on log plot

**Barren plateau**:
```
log(Gradient)
-1 |●
-6 |●●●●●●●●●●●●
   └────────────────→ Epoch
```
Immediate drop to threshold, stays flat

### 3. Comparison Bar Charts

**Error bars interpretation**:
- Small error bars: Consistent performance across seeds
- Large error bars: High variance, unreliable
- Non-overlapping bars: Likely significant difference

---

## Common Result Patterns

### Pattern 1: Clear Winner
```
Approach     | Accuracy | Variance
-------------|----------|----------
Baseline     | 75%      | ±8%
Layerwise    | 93%      | ±2%  ← Clear winner
Local Cost   | 88%      | ±5%
```
**Recommendation**: Use layerwise training

### Pattern 2: Trade-off Scenario
```
Approach     | Accuracy | Time
-------------|----------|------
Baseline     | 85%      | 2m   ← Fast but less accurate
Layerwise    | 92%      | 5m   ← Slow but accurate
Local Cost   | 91%      | 2.5m ← Best balance
```
**Recommendation**: Local cost for efficiency, layerwise for maximum accuracy

### Pattern 3: All Fail (Too Deep)
```
Depth 12:
  Baseline:    0% success
  Layerwise:   0% success
  Local Cost:  0% success
```
**Recommendation**: Reduce circuit depth or use alternative ansatz

---

## Troubleshooting Results

### Problem: Low Accuracy Across All Approaches

**Possible causes**:
1. Data quality issues
2. Inappropriate task for QNN
3. Hyperparameters need tuning
4. Bug in implementation

**Diagnosis steps**:
```python
# Check data
assert X_train.shape == (1000, 16)
assert set(y_train) == {-1, 1}

# Check gradient flow
print(f"Initial gradient: {history['gradient_norms'][0]}")
# Should be > 0.01

# Check loss decrease
assert history['train_loss'][-1] < history['train_loss'][0]
```

### Problem: Barren Plateau Detected

**Indicators**:
- `has_barren_plateau: true`
- Gradient norms < 1e-6 early in training
- Flat loss curve

**Solutions**:
1. **Reduce depth**: Try 2-4 layers instead of 6-8
2. **Use local cost**: Switch to per-qubit measurements
3. **Try layerwise**: Incremental training
4. **Better initialization**: Use pre-training or transfer learning

### Problem: High Variance Across Seeds

**Indicators**:
- Std dev > 10% of mean
- Success rate < 60%

**Solutions**:
1. **More seeds**: Run 10-20 seeds instead of 5
2. **Adjust learning rate**: Try 0.001 or 0.1
3. **Increase epochs**: Allow more training time
4. **Batch size tuning**: Try 10 or 40

---

## Reporting Results

### Minimum Reporting Standards

1. **Accuracy**: Mean ± std across seeds
2. **Success rate**: Percentage achieving threshold
3. **Statistical tests**: p-values and effect sizes
4. **Hyperparameters**: Full configuration details
5. **Computational cost**: Training time and resources

### Example Results Table

| Approach | Depth | Accuracy (%) | Success Rate | Time (s) | BP Rate |
|----------|-------|--------------|--------------|----------|---------|
| Baseline | 4 | 82.3 ± 5.1 | 60% | 145 ± 12 | 40% |
| Baseline | 6 | 68.7 ± 12.3 | 20% | 238 ± 18 | 80% |
| Layerwise | 4 | 91.2 ± 2.4 | 100% | 187 ± 9 | 0% |
| Layerwise | 6 | 89.5 ± 3.8 | 80% | 312 ± 15 | 20% |
| Local Cost | 4 | 89.7 ± 3.2 | 80% | 138 ± 11 | 20% |
| Local Cost | 6 | 87.1 ± 4.5 | 80% | 225 ± 14 | 20% |

---

## Publication-Ready Figures

### Essential Plots

1. **Accuracy comparison** (bar chart with error bars)
2. **Depth impact** (line plot with multiple approaches)
3. **Gradient trajectories** (log-scale time series)
4. **Success rate heatmap** (approach × depth)
5. **Training time analysis** (time vs accuracy scatter)

### Figure Quality Guidelines

- **Resolution**: ≥300 DPI for publication
- **Format**: PDF (vector) or PNG (high-res raster)
- **Fonts**: 10-12 pt labels, 12-14 pt titles
- **Colors**: Color-blind friendly palette
- **Legends**: Clear and concise
- **Axes**: Properly labeled with units

---

## Advanced Analysis

### Learning Curve Analysis

Plot accuracy vs training set size to assess data efficiency:

```python
train_sizes = [100, 200, 500, 1000, 2000]
# Run experiments for each size
# Plot accuracy vs train_size
```

### Hyperparameter Sensitivity

Test robustness to hyperparameter changes:
- Learning rate: [0.001, 0.01, 0.1]
- Batch size: [10, 20, 40]
- Epochs per layer: [5, 10, 20] (layerwise)

### Cross-Validation

For small datasets, use k-fold cross-validation:
```python
for fold in range(5):
    train_idx, test_idx = split_data(fold)
    # Train and evaluate
```

---

## Conclusion Checklist

Before finalizing results:

- [ ] All experiments completed successfully
- [ ] Statistical tests performed
- [ ] Visualizations generated and reviewed
- [ ] Results match expected theoretical behavior
- [ ] Outliers investigated and explained
- [ ] Conclusions supported by data
- [ ] Limitations acknowledged
- [ ] Future work identified
