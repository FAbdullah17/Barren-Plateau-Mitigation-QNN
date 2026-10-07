# Validation Results

This document summarizes configuration validation and the integrity checks performed
before/throughout the production experiment suite. It is split into:

1. **Historical pre-validation** (single-seed, Jan 2026) — the original Go/No-Go sanity check
2. **Current pipeline smoke checks** (8-qubit pipeline)
3. **Production integrity validation** (all 180 runs)

---

## 1. Historical Pre-Validation (Seed 42 Only) — Go/No-Go

Run during the pre-experimentation phase (single seed per config) on the original
4-qubit pipeline.

### Results Summary (Seed 42 Only)

| Approach | 4-Layer | 6-Layer | 8-Layer |
|----------|---------|---------|---------|
| **Baseline** | 76.00% | 76.00% | 76.50% |
| **Layerwise** | 78.00% | 77.00% | 78.00% |
| **Local Cost** | 79.50% | 79.50% | 78.00% |

### Go/No-Go Decision

### ✅ GO for Production Experiments

1. All 9 configurations execute without errors
2. Results save to correct directories
3. Metrics schema is consistent
4. Automation scripts work correctly

> **Note:** An early 5-seed run on the old 4-qubit pipeline showed an 8-layer
> baseline collapse to ~53%. With the current 8-qubit PCA pipeline this is **not
> reproduced** — the baseline reaches ~87% at 8 layers across all 20 seed triples
> (see production results below). That early observation underscores the value of
> the multi-seed, pipeline-controlled suite that eventually replaced it.

---

## 2. Current Pipeline Smoke Checks

Each config also runs a fast smoke test (`*_test.yaml`, 40 updates, 8 qubits) to
verify the end-to-end path on the current pipeline:

| Approach | Updates | Test Acc (seed 0) | Grad-Var |
|----------|---------|-------------------|----------|
| Baseline | 40 | 0.760 | 0.01844 |
| Layerwise | 40 | 0.700 | 0.02010 |
| Local Cost | 40 | 0.580 | 0.00142 |

These confirm the 8-qubit PCA pipeline trains, tracks metrics, and writes
`metrics.json` + `training_history.png` correctly.

---

## 3. Production Integrity Validation (180 Runs)

All 180 production runs (3 approaches × 3 depths × 20 seed triples) were validated:

- **Completeness**: every `seed_{0..19}/metrics.json` present for every approach × depth
- **Schema**: all files match the [Metrics Schema](metrics_schema.md) (`total_updates=2500`,
  `n_parameters` = 64/96/128 by depth, `training_diagnostic` + `history` present)
- **Layout**: one canonical `metrics.json` + `training_history.png` per seed
- **Reproducibility markers**: `data_seed`, `init_seed`, `training_seed` follow the
  base-42 3-step ladder per `seed_index`

Printables:

```bash
python scripts/validate_results.py results/ -v
python scripts/check_output_format.py results/
```

### Production Results vs Historical Validation

| Config | Validation (old seed 42) | Production (20-triple mean) | Notes |
|--------|--------------------------|-----------------------------|-------|
| Baseline 4L | 76.0% | 86.1% ± 3.3% | Pipeline change (PCA/8 qubits) |
| Baseline 6L | 76.0% | 87.2% ± 1.8% | " |
| Baseline 8L | 76.5% | 87.2% ± 2.2% | No collapse in current pipeline |
| Layerwise 4L | 78.0% | 86.4% ± 3.0% | " |
| Layerwise 6L | 77.0% | 86.7% ± 2.4% | " |
| Layerwise 8L | 78.0% | 86.7% ± 2.9% | " |
| Local Cost 4L | 79.5% | 86.1% ± 2.7% | " |
| Local Cost 6L | 79.5% | 87.1% ± 2.2% | " |
| Local Cost 8L | 78.0% | 87.3% ± 2.1% | " |

The 8-point accuracy shift between historical validation and production is explained
by the pipeline upgrade (PCA feature reduction on 8 qubits), not by the seeding scheme.

---

## Files Generated

```
results/
├── baseline/ depth_{4,6,8}/ seed_{0..19}/    (metrics.json, training_history.png)
├── layerwise/ depth_{4,6,8}/ seed_{0..19}/   (metrics.json, training_history.png)
└── local_cost/ depth_{4,6,8}/ seed_{0..19}/  (metrics.json, training_history.png)
```

---

**Status:** COMPLETE — 180 production experiments finished and validated.

**Last Updated:** October 2026