# Developer Assignments for Hybrid-QNN Project

This document provides a complete mapping of who is responsible for which code files and experimental runs.

---

## Phase 1: Setup & Infrastructure (Weeks 1-2)

### Fahad Abdullah - Core Infrastructure
**Files:**
- `src/data/mnist_loader.py` - Data pipeline & quantum encoding
- `src/models/quantum_circuit.py` - Circuit architecture & parameterization
- `src/models/qnn_model.py` - Model architecture & TFQ integration
- `src/utils/optimizer.py` - Optimizer setup

### Asma Zubair - Validation & Monitoring
**Files:**
- `src/evaluation/metrics.py` - Gradient tracking & validation metrics
- `src/models/quantum_circuit.py` - Circuit builder utilities (review)
- `src/data/mnist_loader.py` - Validation framework (review)

### Frahan Riaz - Experiment Framework
**Files:**
- `src/utils/logging_config.py` - Logging & experiment tracking
- `src/utils/checkpointing.py` - Model checkpointing

---

## Phase 2: Implementation (Weeks 3-6)

### Week 3-4: Core Implementation

#### Fahad Abdullah - Baseline Approach
**Files:**
- `src/training/baseline_trainer.py` - Standard end-to-end training implementation
- `experiments/run_baseline.py` - Baseline experiment runner

#### Asma Zubair - Layerwise Approach
**Files:**
- `src/training/layerwise_trainer.py` - Layer-by-layer training implementation
- `experiments/run_layerwise.py` - Layerwise experiment runner

#### Frahan Riaz - Local Cost Approach
**Files:**
- `experiments/run_local_cost.py` - Local cost experiment runner
- (Uses `baseline_trainer.py` with `local_cost=True` flag)

### Week 5-6: Testing & Integration
**All team members** test all three approaches with small-scale runs

---

## Phase 3: Experiments (Weeks 7-9)

### Week 7: 4-Layer Experiments - Asma Zubair
**Configurations (15 runs):**
- `configs/baseline_4layer.yaml` - 5 runs (seeds: 42, 123, 456, 789, 101112)
- `configs/layerwise_4layer.yaml` - 5 runs (seeds: 42, 123, 456, 789, 101112)
- `configs/local_cost_4layer.yaml` - 5 runs (seeds: 42, 123, 456, 789, 101112)

**Expected Results:**
- All approaches should work well at 4 layers (90-95% accuracy)
- Establishes baseline performance

### Week 8: 6-Layer Experiments - Frahan Riaz
**Configurations (15 runs):**
- `configs/baseline_6layer.yaml` - 5 runs (seeds: 42, 123, 456, 789, 101112)
- `configs/layerwise_6layer.yaml` - 5 runs (seeds: 42, 123, 456, 789, 101112)
- `configs/local_cost_6layer.yaml` - 5 runs (seeds: 42, 123, 456, 789, 101112)

**Expected Results:**
- Baseline: Shows degradation (80-85% accuracy)
- Layerwise: Maintains performance (90-93% accuracy)
- Local Cost: Moderate improvement

### Week 9: 8-Layer Experiments - Fahad Abdullah
**Configurations (15 runs):**
- `configs/baseline_8layer.yaml` - 5 runs (seeds: 42, 123, 456, 789, 101112)
- `configs/layerwise_8layer.yaml` - 5 runs (seeds: 42, 123, 456, 789, 101112)
- `configs/local_cost_8layer.yaml` - 5 runs (seeds: 42, 123, 456, 789, 101112)

**Expected Results:**
- **Baseline: <80% accuracy (BARREN PLATEAU - KEY NEGATIVE RESULT)**
- **Layerwise: 90-93% accuracy (SUCCESS - KEY POSITIVE RESULT)**
- Local Cost: Variable performance

**CRITICAL:** These 8-layer results are the core findings of the paper!

---

## Phase 4: Analysis (Week 10)

### Parallel Work on Different Metrics

#### Fahad Abdullah - Gradient Analysis
**Deliverables:**
- Figure 3: Gradient norm trajectories (log scale, all approaches)
- Table 2: Gradient statistics comparison
- Analysis: Barren plateau detection & gradient variance

**Analyzes all 45 runs for gradient behavior**

#### Asma Zubair - Accuracy & Loss Analysis
**Deliverables:**
- Figure 1: Training curves (accuracy over epochs)
- Figure 2: Loss curves (validation loss)
- Table 1: Final test accuracies for all configurations

**Analyzes all 45 runs for performance metrics**

#### Frahan Riaz - Success Rate Analysis
**Deliverables:**
- Figure 4: Success rate bar chart (% runs achieving >90% accuracy)
- Table 3: Training time comparison
- Analysis: Scalability assessment

**Analyzes all 45 runs for success rates and timing**

---

## Phase 5: Paper Writing (Weeks 11-13)

### Week 11: Initial Drafts (Parallel)

#### Fahad Abdullah
**Sections:**
- 1. Introduction
- 2. Background
  - 2.1 Quantum Neural Networks
  - 2.2 Barren Plateau Problem

#### Asma Zubair
**Sections:**
- 3. Methodology
  - 3.1 Experimental Setup
  - 3.2 Training Approaches
  - 3.3 Evaluation Metrics
- 4. Results
  - 4.1 Performance Comparison
  - 4.2 Gradient Analysis

#### Frahan Riaz
**Sections:**
- 4.3 Scalability Analysis
- 5. Discussion
- 6. Conclusion
- References

### Week 12: Review & Revision
**All team members:**
- Cross-review all sections
- Integrate feedback
- Ensure consistency

### Week 13: Finalization
**All team members:**
- Final proofreading
- Format for submission
- Prepare supplementary materials
- Submit to conference

---

## Summary: Workload Distribution

### Experimental Runs (45 total)
- **Asma Zubair:** 15 runs (all 4-layer experiments in Week 7)
- **Frahan Riaz:** 15 runs (all 6-layer experiments in Week 8)
- **Fahad Abdullah:** 15 runs (all 8-layer experiments in Week 9)

### Analysis (Week 10)
- **Fahad:** Gradient analysis on all 45 runs
- **Asma:** Accuracy/loss analysis on all 45 runs
- **Frahan:** Success rate analysis on all 45 runs

### Implementation (Weeks 3-4)
- **Fahad:** Baseline trainer
- **Asma:** Layerwise trainer
- **Frahan:** Local cost experiments (reuses baseline trainer)

### Paper Writing (Weeks 11-13)
- **Fahad:** Introduction & Background (~3 pages)
- **Asma:** Methodology & Results (~4 pages)
- **Frahan:** Discussion & Conclusion (~2 pages)

---

## Quick Reference: File Ownership

### Data & Models (Week 1-2)
```
src/data/mnist_loader.py           → Fahad (Primary), Asma (Review)
src/models/quantum_circuit.py      → Fahad (Primary), Asma (Review)
src/models/qnn_model.py             → Fahad (Primary), Asma (Review)
```

### Training Implementations (Week 3-4)
```
src/training/baseline_trainer.py   → Fahad
src/training/layerwise_trainer.py  → Asma
experiments/run_baseline.py        → Fahad
experiments/run_layerwise.py       → Asma
experiments/run_local_cost.py      → Frahan
```

### Evaluation & Utilities (Week 1-2, Week 10)
```
src/evaluation/metrics.py          → Asma (Primary), Fahad (Support)
src/evaluation/visualization.py    → All (Week 10: Asma-Fig1/2, Fahad-Fig3, Frahan-Fig4)
src/utils/logging_config.py        → Frahan
```

### Configuration Files (Experimental Runs)
```
configs/baseline_4layer.yaml       → Asma (Week 7)
configs/baseline_6layer.yaml       → Frahan (Week 8)
configs/baseline_8layer.yaml       → Fahad (Week 9)

configs/layerwise_4layer.yaml      → Asma (Week 7)
configs/layerwise_6layer.yaml      → Frahan (Week 8)
configs/layerwise_8layer.yaml      → Fahad (Week 9) [KEY FINDING]

configs/local_cost_4layer.yaml     → Asma (Week 7)
configs/local_cost_6layer.yaml     → Frahan (Week 8)
configs/local_cost_8layer.yaml     → Fahad (Week 9)
```

---

## Notes

1. **Parallel Work:** Setup (Weeks 1-2), Implementation (Weeks 3-4), and Analysis (Week 10) involve significant parallel work to meet the 13-week deadline.

2. **Critical Results:** The 8-layer experiments (Week 9, Fahad) are the most important - baseline failure vs. layerwise success.

3. **Daily Communication:** Essential for coordination, especially during parallel phases.

4. **Code Reviews:** All implementations should be reviewed by at least one other team member before experimental runs.

5. **Experiment Tracking:** Use `experiment_tracking.csv` to mark progress on all 45 runs.

---

**Last Updated:** January 2025
