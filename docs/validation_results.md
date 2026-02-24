# Validation Results — Pre-Experimentation Testing

This document summarizes the results of all 9 configuration tests performed during the pre-experimentation validation phase (single seed per config).

---

## Test Environment

- **Date:** January 8, 2026
- **System:** Intel i5-10th Gen, 16GB RAM, CPU-only (no GPU)
- **TensorFlow:** 2.15.0
- **TensorFlow Quantum:** 0.7.2
- **Cirq:** 1.3.0
- **Seed:** 42 (for all validation runs)

---

## Validation Results Summary (Seed 42 Only)

### Accuracy by Depth and Approach

| Approach | 4-Layer | 6-Layer | 8-Layer |
|----------|---------|---------|---------|
| **Baseline** | 76.00% | 76.00% | 76.50% |
| **Layerwise** | 78.00% | 77.00% | 78.00% |
| **Local Cost** | 79.50% | 79.50% | 78.00% |

### Training Time (seconds)

| Approach | 4-Layer | 6-Layer | 8-Layer |
|----------|---------|---------|---------|
| **Baseline** | 638 | 1878 | 2303 |
| **Layerwise** | 391 | 1742 | 3054 |
| **Local Cost** | 1165 | 1629 | 3493 |

### Gradient Statistics

| Config | Mean Gradient Norm | Barren Plateau |
|--------|-------------------|----------------|
| Baseline 4L | 0.279 | No |
| Baseline 6L | 0.276 | No |
| Baseline 8L | 0.224 | No |
| Layerwise 4L | 0.228 | No |
| Layerwise 6L | 0.235 | No |
| Layerwise 8L | 0.226 | No |
| Local Cost 4L | 0.182 | No |
| Local Cost 6L | 0.181 | No |
| Local Cost 8L | 0.175 | No |

---

## Key Findings from Validation

### 1. All Configurations Execute Successfully
All 9 configurations completed training without errors and achieved reasonable accuracy (76-80%).

### 2. Layerwise & Local Cost Outperform Baseline (Single Seed)
- At 4 layers: Local Cost (79.5%) > Layerwise (78%) > Baseline (76.5%)
- At 6 layers: Local Cost (79.5%) > Layerwise (77%) > Baseline (76%)
- At 8 layers: Layerwise = Local Cost (78%) > Baseline (76.5%)

### 3. No Barren Plateau Detected in Validation
With a single seed (42), gradient norms remained healthy (>0.1). This was expected — the barren plateau effect manifests more clearly across multiple seeds where some initializations fall into flat regions.

> **Important:** Production runs with 5 seeds revealed that **8-layer baseline drops to ~53% mean accuracy**, with all 5 seeds failing to exceed 55%. The single-seed validation did not capture this because seed 42 happened to find a relatively favorable initialization. This underscores the importance of multi-seed experiments for studying stochastic phenomena like barren plateaus.

---

## Go/No-Go Decision

### ✅ GO for Production Experiments

**Reasons:**
1. All 9 configurations execute without errors
2. Results save to correct directories
3. Metrics schema is consistent
4. Automation scripts work correctly

---

## Production Results vs Validation

The following table compares single-seed validation results with full 5-seed production results:

| Config | Validation (seed 42) | Production (5-seed mean) | Notes |
|--------|---------------------|--------------------------|-------|
| Baseline 4L | 76.0% | 73.8% | Consistent |
| Baseline 6L | 76.0% | 73.9% | Consistent |
| Baseline 8L | 76.5% | **52.7%** | ⚠️ Barren plateau across seeds |
| Layerwise 4L | 78.0% | 74.0% | Consistent |
| Layerwise 6L | 77.0% | 74.2% | Consistent |
| Layerwise 8L | 78.0% | 73.9% | Consistent |
| Local Cost 4L | 79.5% | 75.3% | Consistent |
| Local Cost 6L | 79.5% | 75.6% | Consistent |
| Local Cost 8L | 78.0% | 75.4% | Consistent |

---

## Files Generated

```
results/
├── baseline/
│   ├── depth_4/seed_42/  ✓
│   ├── depth_6/seed_42/  ✓
│   └── depth_8/seed_42/  ✓
├── layerwise/
│   ├── depth_4/seed_42/  ✓
│   ├── depth_6/seed_42/  ✓
│   └── depth_8/seed_42/  ✓
└── local_cost/
    ├── depth_4/seed_42/  ✓
    ├── depth_6/seed_42/  ✓
    └── depth_8/seed_42/  ✓
```

---

**Validated by:** Pre-experimentation testing
**Date:** January 8, 2026
**Status:** COMPLETE — Production experiments finished

**Last Updated:** February 2026
