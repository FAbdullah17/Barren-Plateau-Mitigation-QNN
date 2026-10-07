# Mitigating Barren Plateaus in Quantum Neural Networks: An Empirical Comparison of Layerwise Training and Local Cost Functions

**Fahad Abdullah, Asma Zubair, Frahan Riaz**

Quantum Machine Learning Research Initiative

---

## Abstract

The barren plateau phenomenon, in which the gradients of a variational quantum circuit vanish exponentially with circuit size, is a central obstacle to scaling variational quantum algorithms for machine learning (McClean et al., 2018). Two mitigation strategies — layerwise training (Skolik et al., 2020) and local cost functions (Cerezo et al., 2021) — have been proposed independently, but no direct, controlled comparison existed on a standardized benchmark. We present a systematic empirical study of both strategies against an end-to-end baseline on binary MNIST classification (digits 3 vs. 6) using an 8-qubit hardware-efficient ansatz across circuit depths of 4, 6, and 8 layers. Each of the nine approach-by-depth conditions was replicated over 20 randomized seed triples (180 runs in total). All three strategies reach statistically indistinguishable classification accuracy (~86–87%) at every depth, so the barren-plateau signature is not observable in accuracy at this scale. The differentiating signal appears in the gradient statistics: the local cost function maintains 1.5–2.4× lower per-parameter gradient variance than the global-cost baseline at all depths (Welch t-test, p ≤ 0.007), at no cost in accuracy. Layerwise training is qualitatively distinct in that its gradient variance decreases through training, whereas baseline and local-cost training show increasing variance. These results provide the first standardized, reproducible comparison of the two leading mitigation approaches and identify gradient variance — not accuracy — as the informative diagnostic at moderate qubit counts.

---

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Background](#2-background)
- [3. Methods](#3-methods)
- [4. Results](#4-results)
- [5. Discussion](#5-discussion)
- [6. Conclusion and Future Work](#6-conclusion-and-future-work)
- [References](#references)
- [Appendix A. Reproducibility](#appendix-a-reproducibility)
- [Appendix B. Supplementary Material and Results Inventory](#appendix-b-supplementary-material-and-results-inventory)

---

## 1. Introduction

Variational quantum algorithms (VQAs) train parameterized quantum circuits by classical optimization of a cost function evaluated on the quantum device. Quantum neural networks (QNNs) — VQAs applied to machine learning tasks — are among the most promising near-term applications of quantum computing (Broughton et al., 2020). A fundamental barrier to scaling QNNs is the *barren plateau* phenomenon: as the number of qubits or the circuit depth grows, the variance of the cost-function gradients decreases exponentially over the parameter space, making the landscape effectively flat and gradient-based training impossible (McCallen, 2018). A single-qubit observable (a "local" cost) is known to concentrate more slowly than a many-body observable (a "global" cost), connecting cost-function locality to trainability (Cerezo et al., 2021). Two mitigation strategies dominate the literature:

- **Layerwise training** (Skolik et al., 2020): build and optimize the circuit incrementally, one layer at a time, freezing previously trained layers and optionally fine-tuning at the end. Each stage optimizes in a smaller parameter space, avoiding the flat landscape of a fully randomized deep circuit.
- **Local cost functions** (Cerezo et al., 2021): replace a global observable with a sum of per-qubit operators, so that gradient variance scales polynomially rather than exponentially with the number of qubits.

**Research question.** How do layerwise training and local cost functions compare in their ability to mitigate barren plateaus in QNNs, as measured by classification accuracy, gradient statistics, training dynamics, and computational cost?

**Contributions.**

1. A controlled, reproducible comparison of both strategies against an end-to-end baseline spanning three circuit depths and 20 seed triples per condition (180 runs).
2. An explicit treatment of gradient variance — the theoretically relevant barren-plateau statistic — alongside accuracy, showing that the two objectives separate cleanly.
3. A fully open implementation (TensorFlow Quantum), configuration, and result corpus enabling independent replication.

---

## 2. Background

### 2.1 Barren Plateaus

For a parameterized circuit $U(\theta)$ with cost $C(\theta)$, the gradient-component variance of a random initialization obeys

$$\text{Var}\left[\frac{\partial C}{\partial \theta_i}\right] \leq F(n),\qquad F(n) \in \mathcal{O}\left(\frac{1}{b^n}\right),\; b > 1,$$

i.e., gradients vanish exponentially in the number of qubits $n$(McClean et al., 2018). Consequences include flat loss landscapes, stagnated optimization, and accuracy near random guessing in classification. The concentration is cost-function dependent: a global observable (coupling many qubits) concentrates exponentially with depth even in shallow circuits, whereas local observables train polynomially (Cerezo et al., 2021).

### 2.2 Layerwise Training

Incremental training proceeds by (i) optimizing a shallow 1-layer circuit, (ii) freezing its parameters, (iii) adding and optimizing the next layer, and (iv) optionally fine-tuning the full circuit. Because each stage optimizes obtainable, rather than randomized, parameters, effective depth grows gradually and the landscape remains trainable. In our implementation the total update budget of 2,500 gradient steps is preserved across approaches: it is split into `per_stage` updates per layer plus a `finetune` phase (Skolik et al., 2020; Appendix A).

### 2.3 Local Cost Functions

Under a global observable $Z_0$ on qubit 0, gradient variance decays as $1/2^n$. Replacing the readout with $\sum_i Z_i$ (one Pauli-$Z$ per qubit) yields variance scaling as $1/\text{poly}(n)$. Our local-cost configuration measures every qubit individually and averages a loss over the per-qubit expectations; the baseline measures a single qubit (Cerezo et al., 2021).

### 2.4 Hardware-Efficient Ansatz

We use a hardware-efficient ansatz (Kandala et al., 2017): each layer applies an $RY(\theta)$ then $RZ(\theta)$ rotation to every qubit, followed by a linear (nearest-neighbor) CNOT ladder. This gives $2 \times n_\text{qubits} \times n_\text{layers}$ trainable parameters — 64, 96, and 128 at depths 4, 6, and 8.

---

## 3. Methods

### 3.1 Dataset and Preprocessing

We use the MNIST binary task (digits 3 vs. 6) with 1,000 training and 200 test samples. Full-resolution 28×28 images (784 dims) are reduced to a fixed low-dimensional input in four steps: bilinear downsample to 4×4 (16 features), principal-component analysis (PCA) to the 8 leading components (fit on the training split only), min-max normalization using training-only statistics, and RY angle encoding $\theta_i = x_i \cdot \pi$ on 8 qubits. No test information is used in any transform.

### 3.2 Quantum Circuit

Each sample is encoded by single-qubit $RY(x_i\pi)$ gates into an 8-qubit register, followed by an $L$-layer hardware-efficient ansatz (Section 2.4). Readout is the expectation of $Z$ on qubit 0 (baseline and layerwise; "global") or of $Z$ on every qubit (local cost). All experiments use state-vector simulation through TensorFlow Quantum 0.7.2, TensorFlow 2.15, and Cirq 1.3 on Python 3.10.

### 3.3 Training Protocols

All configurations use the Adam optimizer (learning rate 0.01, batch size 20) and binary cross-entropy loss over 2,500 gradient updates:

| Approach | Protocol |
|----------|----------|
| **Baseline** | Full circuit trained end-to-end; global readout. |
| **Layerwise** | Global readout; layered training with total budget preserved: 500 per stage + 500 fine-tune (4L), 357 per stage + 358 fine-tune (6L), 277 per stage + 284 fine-tune (8L). |
| **Local cost** | End-to-end training as baseline with per-qubit ($Z_i$) readout. |

### 3.4 Gradient Diagnostic

Barren-plateau severity is quantified by the mean per-parameter gradient variance

$$\bar{V}^{x} = \frac{1}{P}\sum_{p=1}^{P}\text{Var}_x\left[\frac{\partial \ell}{\partial \theta_p}\right],$$

the mean over parameters of the variance-over-samples of each parameter's gradient, sampled every 10 updates and logged as a trajectory throughout training. We report the final value and the trajectory trend. No binary threshold-based claim is used.

### 3.5 Experimental Design

| | 4 Layers | 6 Layers | 8 Layers |
|--|----------|----------|----------|
| **Baseline** | 20 seeds | 20 seeds | 20 seeds |
| **Layerwise** | 20 seeds | 20 seeds | 20 seeds |
| **Local Cost** | 20 seeds | 20 seeds | 20 seeds |

Total: **180 runs** (3 approaches × 3 depths × 20 seed triples). Each run consumes one seed triple $(s, s{+}1, s{+}2)$ with $s = 42 + 3i$ for seed index $i = 0, \dots, 19$, decoupling data subsampling, parameter initialization, and batch ordering. Shared hyperparameters: 8 qubits, 1,000/200 train/test split, learning rate 0.01, batch 20, 2,500 updates, PCA → 8 components, RY angle encoding, binary cross-entropy.

### 3.6 Statistical Procedures

Each approach-by-depth cell is summarized by mean ± std across the 20 seed triples. Pairwise comparisons use Welch's t-test (unequal variances) on both test accuracy and final gradient variance; effect sizes for the gradient-variance contrast are reported as Cohen's $d$. Significance is assessed at $\alpha = 0.05$ (two-sided).

---

## 4. Results

All quantities below are computed from the full 180-run corpus. Cell statistics are mean ± std over 20 seed triples.

### 4.1 Classification Accuracy

| Approach | 4 Layers | 6 Layers | 8 Layers |
|----------|----------|----------|----------|
| Baseline | 86.1% ± 3.3% | 87.2% ± 1.8% | 87.2% ± 2.2% |
| Layerwise | 86.4% ± 3.0% | 86.7% ± 2.4% | 86.7% ± 2.9% |
| Local Cost | 86.1% ± 2.7% | 87.1% ± 2.2% | 87.3% ± 2.1% |

Observed per-run ranges (4/6/8L): baseline 79.0/84.0/82.5% to 90.5/91.0/91.0%; layerwise 80.5/82.5/78.5% to 91.0/91.5/91.5%; local cost 80.5/81.5/83.0% to 90.5/90.5/91.5%. Accuracy is statistically indistinguishable between approaches at every depth (all pairwise Welch p ≥ 0.43); no approach exhibits depth degradation. Notably, the baseline does **not** collapse at 8 layers at this scale.

### 4.2 Gradient Variance

| Config | Final Grad-Var | Std | vs. Baseline |
|--------|----------------|-----|--------------|
| Baseline 4L | 0.00694 | 0.00225 | 1.00× |
| Baseline 6L | 0.00658 | 0.00174 | 1.00× |
| Baseline 8L | 0.00350 | 0.00146 | 1.00× |
| Layerwise 4L | 0.00731 | 0.00195 | 1.05× |
| Layerwise 6L | 0.00628 | 0.00167 | 0.96× |
| Layerwise 8L | 0.00452 | 0.00089 | 1.29× |
| **Local Cost 4L** | **0.00319** | 0.00146 | **0.46× (2.2× lower)** |
| **Local Cost 6L** | **0.00279** | 0.00079 | **0.42× (2.4× lower)** |
| **Local Cost 8L** | **0.00240** | 0.00075 | **0.69× (1.5× lower)** |

The local cost function yields significantly lower gradient variance than the baseline at every depth (p = 7.8e-7, 3.5e-9, 6.8e-3 for 4/6/8L) and than layerwise training (p = 1.2e-8, 7.3e-9, 1.8e-9), with large effect sizes throughout (Cohen's d = 1.9, 2.7, 0.9 for 4/6/8L). This is the strongest, cleanest mitigation signal in the corpus.

### 4.3 Gradient Trajectories

| Approach | Dominant trajectory | Runs with rising variance |
|----------|---------------------|---------------------------|
| Baseline | Rises during training | 16–19/20 |
| Layerwise | Falls during training | 0–1/20 |
| Local Cost | Rises mildly | 14/20 |

Layerwise training is the only approach whose gradient variance decreases as training proceeds — a structurally distinct optimization path consistent with its incremental schedule. Baseline and local-cost training both exhibit modestly increasing gradient variance.

### 4.4 Success Rate

At a ≥70% accuracy threshold, all approaches succeed on all 20 runs at every depth (100%). At a stricter ≥85% threshold (4/6/8L): baseline 75/90/90%, layerwise 60/75/85%, local cost 60/85/90%. Approach separation is weak and consistent with the accuracy analysis.

### 4.5 Training Time

| Approach | 4L | 6L | 8L |
|----------|----|----|----|
| Baseline | 4.2 h | 1.7 h | 1.7 h |
| Layerwise | 1.1 h | 0.7 h | 1.8 h |
| Local Cost | 3.5 h | 4.1 h | 6.6 h |

Means over 20 runs per cell (CPU, state-vector simulation). Local cost is the most expensive at depth; layerwise is the cheapest for shallow circuits. The full 180-run suite totals approximately 506 CPU-hours (≈21 CPU-days) summed across runs.

---

## 5. Discussion

**Accuracy does not separate the strategies.** At 8 qubits, all approaches — including the unmitigated baseline — train successfully to ~86–87% at every depth. The previously reported catastrophic 8-layer baseline collapse observed on a smaller-scale pipeline is not reproduced, indicating that at this scale the circuit is not deep enough for unconditional barren-plateau saturation. Consequently, classification accuracy is an insensitive diagnostic at moderate qubit counts.

**Gradient variance is the informative signal.** The only statistically robust effect is the local cost function's reproducible 1.5–2.4× reduction in per-parameter gradient variance, achieved at no penalty in accuracy or stability. This matches the theoretical prediction that local observables avoid exponential gradient concentration (Cerezo et al., 2021) and identifies gradient variance as the appropriate empirical probe for mitigation effectiveness.

**Layerwise training behaves differently in kind, not degree.** Its falling gradient-variance trajectory marks a distinct optimization path. However, this qualitative difference does not translate into lower final variance — layerwise final variance is comparable to (or slightly above) the baseline — suggesting its benefits are procedural (training stability) rather than statistical here.

**Limitations.** Single benchmark and digit pair; 8-qubit, noiseless state-vector simulation only; accuracy ceiling likely bounded by the 4×4-downsample + PCA feature budget; ~500 CPU-hours of computational cost limits breadth of the depth sweep.

---

## 6. Conclusion and Future Work

We provide the first controlled, 180-run comparison of layerwise training and local cost functions for barren-plateau mitigation on a standardized QNN benchmark. At 8 qubits, both strategies preserve trainability and accuracy at depth, and local cost functions deliver a statistically significant, reproducibility-verified reduction in gradient variance — the theoretically relevant statistic — without accuracy loss. Future work includes deeper circuits (10–16 layers) to map the barrier onset curve, noisy and hardware-realistic simulation, hybrid layerwise+local-cost strategies, multi-class tasks, and a comparison of circuit expressibility against trainability.

---

## References

1. McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., & Neven, H. (2018). Barren plateaus in quantum neural network training landscapes. *Nature Communications*, 9(1), 4812. [DOI: 10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4)
2. Skolik, A., McClean, J. R., Mohseni, M., van der Smagt, P., & Leib, M. (2020). Layerwise learning for quantum neural networks. *Quantum Machine Intelligence*, 3(1), 5. [arXiv: 2006.14904](https://arxiv.org/abs/2006.14904)
3. Cerezo, M., Sone, A., Volkoff, T., Cincio, L., & Coles, P. J. (2021). Cost function dependent barren plateaus in shallow parametrized quantum circuits. *Nature Communications*, 12(1), 1791. [DOI: 10.1038/s41467-021-21728-w](https://doi.org/10.1038/s41467-021-21728-w)
4. Kandala, A., Mezzacapo, A., Temme, K., Takita, M., Brink, M., Chow, J. M., & Gambetta, J. M. (2017). Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets. *Nature*, 549(7671), 242–246. [DOI: 10.1038/nature23879](https://doi.org/10.1038/nature23879)
5. Broughton, M., et al. (2020). TensorFlow Quantum: A software framework for quantum machine learning. [arXiv: 2003.02989](https://arxiv.org/abs/2003.02989)

---

## Appendix A. Reproducibility

### A.1 Environment

Python 3.10 (required by TensorFlow Quantum 0.7.2), TensorFlow 2.15.0, TensorFlow Quantum 0.7.2, Cirq 1.3.0, NumPy, SciPy, scikit-learn, Matplotlib. All pinned dependencies are in `requirements.txt`. CPU-only execution is sufficient; ≥8 GB RAM recommended.

```bash
git clone https://github.com/FAbdullah17/Barren-Plateau-Mitigation-QNN.git
cd Barren-Plateau-Mitigation-QNN
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### A.2 Running Experiments

Each run is driven by a YAML configuration and a seed index; the seed triple is derived as described in Section 3.5.

```bash
python experiments/run_baseline.py    configs/baseline_4layer.yaml  --seed-index 0
python experiments/run_layerwise.py   configs/layerwise_6layer.yaml --seed-index 5
python experiments/run_local_cost.py  configs/local_cost_8layer.yaml --seed-index 10
```

Batch runners execute all 20 seeds for a depth (`scripts/run_4layer_experiments.py`, `run_6layer_experiments.py`, `run_8layer_experiments.py`) or for one approach/depth (`scripts/run_batch.py`). Full reproduction of the 180-run suite takes ~500 CPU-hours sequentially. A quick smoke test uses the `*_test.yaml` configs (40 updates).

### A.3 Configuration Schema

```yaml
experiment:
  name: "baseline_4layer"
  approach: "baseline"
model:
  n_qubits: 8
  n_layers: 4
training:
  optimizer: "adam"
  learning_rate: 0.01
  batch_size: 20
  total_updates: 2500
  cost_function: "global"       # global | local
data:
  dataset: "mnist"
  digit1: 3
  digit2: 6
  train_size: 1000
  test_size: 200
  image_size: [4, 4]
  preprocessing: "pca"
  n_components: 8
  encoding: "ry_angle"
seeds:
  seed_triples: 20
  base_seed: 42
metrics:
  track_gradients: true
  log_frequency: 10
output:
  results_dir: "results/baseline/depth_4"
  save_plot: true
  checkpoint_frequency: 500
```

### A.4 Results Layout and Validation

Each run writes `metrics.json` and `training_history.png` to `results/<approach>/depth_<L>/seed_<S>/`. The schema is specified in [`docs/metrics_schema.md`](docs/metrics_schema.md); corpus integrity is checked with:

```bash
python scripts/validate_results.py results/ -v
python scripts/check_output_format.py results/
```

---

## Appendix B. Supplementary Material and Results Inventory

| Document | Content |
|----------|---------|
| [`docs/methodology.md`](docs/methodology.md) | Detailed derivations, algorithms, and training budgets |
| [`docs/results_report.md`](docs/results_report.md) | Full results report (all tables and statistics) |
| [`docs/results_interpretation.md`](docs/results_interpretation.md) | Statistical analysis, effect sizes, and interpretation |
| [`docs/quickstart.md`](docs/quickstart.md) | Reproduction quick start |
| [`docs/metrics_schema.md`](docs/metrics_schema.md) | Output schema specification |
| [`docs/validation_results.md`](docs/validation_results.md) | Pre-experiment validation and smoke checks |
| [`notebooks/`](notebooks/) | Data, circuit, and results analysis notebooks |

Licensed under the [MIT License](LICENSE). If you find this work useful for research, please consider citing it.