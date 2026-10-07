# Results Report — 180 Production Runs

**Date:** October 2026
**Suite:** 3 approaches × 3 depths (4, 6, 8) × 20 seed triples = **180 completed runs**

---

## 1. Experimental Setup (Summary)

| Setting | Value |
|---------|-------|
| Dataset | MNIST binary (3 vs 6), 1000 train / 200 test |
| Preprocessing | Downsample 4×4 → PCA → 8 components |
| Encoding | RY angle on 8 qubits |
| Ansatz | Hardware-efficient: RY+RZ per qubit + linear CNOT ladder |
| Parameters | 64 / 96 / 128 for depths 4 / 6 / 8 |
| Optimizer | Adam, lr 0.01, batch 20 |
| Updates | 2500 per run (layerwise split into stages + fine-tune) |
| Seed triples | 20 per condition: `(42+3i, 43+3i, 44+3i)`, i = 0..19 |

---

## 2. Integrity Validation

- **180/180** metrics present, one per seed, canonical layout (`seed_{0..19}/metrics.json`)
- Schema consistent: `total_updates=2500`, correct `n_parameters` per depth, `training_diagnostic` + `history` + `pca_info` present
- Layout clean: exactly one `metrics.json` + `training_history.png` per seed, no duplicates or stray files

---

## 3. Accuracy Results (Test Acc)

### Mean ± Std (%)

| Approach | 4 Layers | 6 Layers | 8 Layers |
|----------|----------|----------|----------|
| Baseline | 86.1 ± 3.3 | 87.2 ± 1.8 | 87.2 ± 2.2 |
| Layerwise | 86.4 ± 3.0 | 86.7 ± 2.4 | 86.7 ± 2.9 |
| Local Cost | 86.1 ± 2.7 | 87.1 ± 2.2 | 87.3 ± 2.1 |

### Ranges (%)

| Approach | Depth | Min | Max |
|----------|-------|-----|-----|
| Baseline | 4 / 6 / 8 | 79.0 / 84.0 / 82.5 | 90.5 / 91.0 / 91.0 |
| Layerwise | 4 / 6 / 8 | 80.5 / 82.5 / 78.5 | 91.0 / 91.5 / 91.5 |
| Local Cost | 4 / 6 / 8 | 80.5 / 81.5 / 83.0 | 90.5 / 90.5 / 91.5 |

**Reading:** Accuracy is flat (~86-87%) across depth for every approach. Pairwise Welch
t-tests give p ≥ 0.43 at all depths — accuracy does not separate the strategies.

---

## 4. Gradient Variance (Primary Diagnostic)

Final `mean_param_grad_variance` (V̄ˣ), mean across seeds:

| Approach | 4L | 6L | 8L |
|----------|------|------|------|
| Baseline | 0.00694 | 0.00658 | 0.00350 |
| Layerwise | 0.00731 | 0.00628 | 0.00452 |
| Local Cost | **0.00319** | **0.00279** | **0.00240** |

### Reduction vs Baseline

| Depth | Local/Baseline | Welch p |
|-------|----------------|---------|
| 4L | 0.46× (2.2× lower) | 7.8e-7 |
| 6L | 0.42× (2.4× lower) | 3.5e-9 |
| 8L | 0.69× (1.5× lower) | 6.8e-3 |

Local cost also beats layerwise at every depth (p = 1.2e-8, 7.3e-9, 1.8e-9).

### Trajectory Trends (grad-variance rises during training)

| Approach | Rising runs |
|----------|-------------|
| Baseline | 16-19/20 |
| Layerwise | 0-1/20 |
| Local Cost | 14/20 |

Layerwise is the only approach whose gradient variance **falls** during training — a
distinct optimization signature.

---

## 5. Success Rates

**≥70% threshold: 100% (20/20) for every approach × depth.**

**≥85% threshold (4/6/8L):** Baseline 75/90/90%, Layerwise 60/75/85%, Local Cost 60/85/90%.

---

## 6. Training Time (mean, hours)

| Approach | 4L | 6L | 8L |
|----------|-----|-----|-----|
| Baseline | 4.23 | 1.67 | 1.66 |
| Layerwise | 1.06 | 0.70 | 1.79 |
| Local Cost | 3.47 | 4.13 | 6.57 |

Local cost is costliest at depth; layerwise is cheapest for shallow circuits.

---

## 7. Conclusions

1. **No barren-plateau collapse at this scale** — baseline reaches ~87% at all depths; the earlier 8L collapse observed on the 4-qubit pipeline is not reproduced.
2. **Accuracy is statistically tied** across approaches (p ≥ 0.43).
3. **Local cost reproducibly lowers gradient variance** by 1.5-2.4× (p ≤ 0.007) at all depths, with no accuracy penalty — the strongest, cleanest mitigation signal in this suite.
4. **Layerwise training is qualitatively distinct** — its gradient-variance trajectory decreases through training.

---

## 8. Caveats

- Single benchmark (MNIST 3-vs-6), 8 qubits, state-vector simulation (no noise)
- Gradient variance measured from simulation; hardware realizations may differ
- Accuracy ceiling likely limited by the 4×4 downsampling + PCA feature budget

---

Generated from the `results/` metrics (`scripts/validate_results.py`, `scripts/check_output_format.py`).