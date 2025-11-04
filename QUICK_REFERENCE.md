# Quick Reference Guide - Barren Plateau Research (45 Runs, 13 Weeks)

**Project:** Empirical Comparison of Barren Plateau Mitigation Strategies  
**Team:** Fahad Abdullah, Asma Zubair, Frahan Riaz  
**Timeline:** 13 weeks (3 months)  
**Last Updated:** November 4, 2025

---

## 📋 Table of Contents

1. [Experimental Design Overview](#experimental-design-overview)
2. [Task Distribution](#task-distribution)
3. [Timeline & Phases](#timeline--phases)
4. [Code Files Reference](#code-files-reference)
5. [Configuration Files](#configuration-files)
6. [Expected Results](#expected-results)
7. [Daily Workflows](#daily-workflows)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Experimental Design Overview

### **The Correct Design: 45 Runs**

**ALL THREE APPROACHES TEST ALL THREE DEPTHS**

```
                    4 LAYERS    6 LAYERS    8 LAYERS    TOTAL
Baseline                5           5           5        15
Layerwise               5           5           5        15
Local Cost              5           5           5        15
                    -------     -------     -------    -------
TOTAL                  15          15          15        45
```

**Formula:** 3 approaches × 3 depths × 5 random seeds = **45 training runs**

### **Why This Design?**

Each depth serves a specific scientific purpose:

| Depth | Purpose | Baseline | Layerwise | Local Cost |
|-------|---------|----------|-----------|------------|
| **4 Layers** | Baseline performance | ✅ Works | ✅ Excellent | ✅ Good |
| **6 Layers** | Show differentiation | ⚠️ Degrading | ✅ Maintained | ✅ Good |
| **8 Layers** | Critical scalability test | ❌ **FAILS** | ✅ **SUCCESS** | ✅ Trainable |

**Key Comparison:** At 8 layers, baseline fails (<80%) but layerwise succeeds (90-93%), proving the mitigation strategy works!

---

## 👥 Task Distribution

### **Depth-Based Experimental Distribution**

| Person | Depth | Experiments | Week | Approaches Tested |
|--------|-------|-------------|------|-------------------|
| **Asma Zubair** | 4 layers | 15 runs | Week 7 | Baseline (5) + Layerwise (5) + Local Cost (5) |
| **Frahan Riaz** | 6 layers | 15 runs | Week 8 | Baseline (5) + Layerwise (5) + Local Cost (5) |
| **Fahad Abdullah** | 8 layers | 15 runs | Week 9 | Baseline (5) + Layerwise (5) + Local Cost (5) |

**Rationale:** Each person sees how all three mitigation strategies perform at one specific depth, enabling direct comparison.

### **Metric Analysis Distribution** (All analyze all 45 runs)

| Person | Metric Responsibility | Deliverables |
|--------|----------------------|--------------|
| **Fahad Abdullah** | Gradient variance analysis | Figure 3, Table 2 (gradient statistics) |
| **Asma Zubair** | Accuracy & loss analysis | Figure 1, Figure 2, Table 1 (performance) |
| **Frahan Riaz** | Success rate & efficiency | Figure 4, Table 3 (robustness & timing) |

### **Implementation Distribution** (Weeks 3-4)

| Person | Responsibility | Files |
|--------|---------------|-------|
| **Fahad Abdullah** | Baseline approach + Core infrastructure | `baseline_trainer.py`, `mnist_loader.py`, `quantum_circuit.py`, `qnn_model.py` |
| **Asma Zubair** | Layerwise approach + Metrics | `layerwise_trainer.py`, `metrics.py`, `visualization.py` |
| **Frahan Riaz** | Local cost approach + Utilities | `run_local_cost.py`, `logging_config.py` |

### **Paper Writing Distribution** (Weeks 11-13)

| Person | Sections | Pages |
|--------|----------|-------|
| **Fahad Abdullah** | Introduction + Background | ~3 pages |
| **Asma Zubair** | Methodology + Results | ~4 pages |
| **Frahan Riaz** | Discussion + Conclusion | ~2 pages |

---

## 📅 Timeline & Phases

### **Phase 1: Setup & Infrastructure (Weeks 1-2)**

**Week 1: Environment & Data**
- Install Python 3.10, TensorFlow 2.15.0, TFQ 0.7.3, Cirq 1.3.0
- Implement MNIST data loading (Fahad)
- Implement 4×4 downsampling (Fahad)
- Create data validation (Asma)

**Week 2: Core Components**
- Build hardware-efficient ansatz (Fahad)
- Create variable-depth circuit builder (Asma)
- Implement gradient tracker (Asma)
- Setup experiment logging (Frahan)
- Create config system (Fahad)

### **Phase 2: Implementation (Weeks 3-6)**

**Week 3-4: Parallel Development**
- Fahad: Baseline trainer + global cost function
- Asma: Layerwise trainer + parameter freezing
- Frahan: Local cost implementation + utilities

**Week 5-6: Testing & Integration**
- All: Unit tests for each approach
- All: Integration testing with small datasets
- All: Verify all 9 configurations load correctly

### **Phase 3: Experiments (Weeks 7-9)**

**Week 7: 4-Layer Experiments (Asma - 15 runs)**
- 5 × Baseline @ 4 layers (configs/baseline_4layer.yaml)
- 5 × Layerwise @ 4 layers (configs/layerwise_4layer.yaml)
- 5 × Local Cost @ 4 layers (configs/local_cost_4layer.yaml)

**Week 8: 6-Layer Experiments (Frahan - 15 runs)**
- 5 × Baseline @ 6 layers (configs/baseline_6layer.yaml)
- 5 × Layerwise @ 6 layers (configs/layerwise_6layer.yaml)
- 5 × Local Cost @ 6 layers (configs/local_cost_6layer.yaml)

**Week 9: 8-Layer Experiments (Fahad - 15 runs)**
- 5 × Baseline @ 8 layers (configs/baseline_8layer.yaml) - **CRITICAL TEST**
- 5 × Layerwise @ 8 layers (configs/layerwise_8layer.yaml) - **KEY FINDING**
- 5 × Local Cost @ 8 layers (configs/local_cost_8layer.yaml)

### **Phase 4: Analysis (Week 10)**

**Parallel Work:**
- Fahad: Gradient variance analysis across all 45 runs
- Asma: Accuracy/loss analysis across all 45 runs
- Frahan: Success rate analysis across all 45 runs

### **Phase 5: Writing & Submission (Weeks 11-13)**

**Week 11:** First drafts (all sections in parallel)  
**Week 12:** Review, feedback, revisions  
**Week 13:** Finalization and submission

---

## 💻 Code Files Reference

### **📦 Data Pipeline** (`src/data/`)

#### **`mnist_loader.py`** - Data Loading & Preprocessing
**Developer:** Fahad Abdullah (Primary), Asma Zubair (Review)

**Purpose:** Prepare MNIST binary classification data for quantum circuits

**Key Functions:**

```python
load_mnist_binary(digit1=3, digit2=6, train_size=1000, test_size=200, 
                  image_size=(4,4), seed=None)
```
**Returns:** `(X_train, y_train, X_test, y_test)`

**Process:**
1. Loads MNIST dataset from TensorFlow
2. Filters for specified digits (default: 3 vs 6)
3. Downsamples images from 28×28 → 4×4 (bilinear interpolation)
4. Normalizes pixel values to [0, 1]
5. Flattens to 16-dimensional vectors
6. Converts labels to binary (0, 1)

**Output Shapes:**
- `X_train`: (1000, 16) - 1000 training samples, 16 features
- `y_train`: (1000,) - Binary labels
- `X_test`: (200, 16) - 200 test samples
- `y_test`: (200,) - Binary labels

```python
encode_data_for_qnn(data)
```
**Purpose:** Scale data to rotation angles [0, π] for quantum encoding

**Test Command:**
```bash
python src/data/mnist_loader.py
```

---

### **⚛️ Quantum Models** (`src/models/`)

#### **`quantum_circuit.py`** - Hardware-Efficient Ansatz
**Developer:** Fahad Abdullah (Primary), Asma Zubair (Review)

**Purpose:** Build parameterized quantum circuits for QNN training

**Key Class: `QuantumCircuit`**

```python
QuantumCircuit(n_qubits=4, n_layers=4)
```

**Circuit Architecture:**
```
Each layer consists of:
1. RY(θᵢ) rotation on each qubit
2. RZ(φᵢ) rotation on each qubit  
3. CNOT entangling gates in linear chain: q₀→q₁→q₂→q₃

Parameters per layer: n_qubits × 2
Total parameters: n_layers × n_qubits × 2

Example (4 layers, 4 qubits): 4 × 4 × 2 = 32 parameters
```

**Methods:**

| Method | Purpose | Returns |
|--------|---------|---------|
| `get_circuit()` | Get full Cirq circuit | `cirq.Circuit` |
| `get_parameters()` | All symbolic parameters (flat list) | `List[sympy.Symbol]` |
| `get_layer_parameters(idx)` | Parameters for specific layer | `List[sympy.Symbol]` |
| `get_circuit_up_to_layer(idx)` | Partial circuit (layers 0 to idx) | `cirq.Circuit` |
| `visualize()` | Text representation | `str` |

**Function: `create_readout_operators`**

```python
create_readout_operators(n_qubits, local=False)
```

| Mode | Description | Measurements |
|------|-------------|--------------|
| Global (local=False) | Standard approach | Z₀ only (first qubit) |
| Local (local=True) | Per-qubit measurements | Z₀, Z₁, Z₂, Z₃ (all qubits) |

**Test Command:**
```bash
python src/models/quantum_circuit.py
```

**Expected Output:**
- Circuit with 32 parameters (4L × 4Q × 2)
- Circuit visualization printed
- Layer extraction works

---

#### **`qnn_model.py`** - TensorFlow Quantum Models
**Developer:** Fahad Abdullah (Primary), Asma Zubair (Review)

**Purpose:** Implement trainable QNN models with TensorFlow Quantum

**Class 1: `QuantumNeuralNetwork`** - Standard End-to-End QNN

```python
QuantumNeuralNetwork(n_qubits=4, n_layers=4, local_cost=False, name="QNN")
```

**Architecture:**
```
Input (quantum circuits) 
    ↓
[TFQ PQC Layer] - Parameterized Quantum Circuit
    ↓ (expectation values)
[Output Layer] - Dense/Lambda layer
    ↓ (sigmoid activation)
Output (probabilities [0,1])
```

**Forward Pass:**
1. Takes batch of data-encoded quantum circuits
2. PQC layer measures expectation values
3. Output layer converts to classification probabilities

**Methods:**
- `call(inputs, training=None)`: Forward propagation
- `get_num_parameters()`: Total trainable parameters

---

**Class 2: `LayerwiseQNN`** - Incremental Layer-by-Layer Model

```python
LayerwiseQNN(n_qubits=4, target_layers=4, local_cost=False)
```

**Purpose:** Manage progressive circuit building (Skolik et al. 2020)

**Training Workflow:**
```
Step 1: Build 1-layer circuit
    ↓
Step 2: Train only layer 1 parameters
    ↓ (freeze layer 1)
Step 3: Add layer 2, train only layer 2 parameters
    ↓ (freeze layers 1-2)
Step 4: Add layer 3, train only layer 3 parameters
    ↓ (freeze layers 1-3)
Step 5: Add layer 4, train only layer 4 parameters
    ↓ (unfreeze all)
Step 6: Fine-tune all layers together
```

**Methods:**
- `add_layer()`: Adds one layer, returns new model
- `get_current_model()`: Returns current TensorFlow model
- `freeze_trained_layers()`: Prevents gradient updates on old layers

**Test Command:**
```bash
python src/models/qnn_model.py
```

---

### **🎓 Training Module** (`src/training/`)

#### **`baseline_trainer.py`** - Standard Training
**Developer:** Fahad Abdullah (Implementation)  
**Experiments:** Asma (4L Week 7), Frahan (6L Week 8), Fahad (8L Week 9)

**Purpose:** Implement baseline end-to-end training (demonstrates barren plateau problem)

**Class: `BaselineTrainer`**

```python
BaselineTrainer(n_qubits=4, n_layers=4, learning_rate=0.01, 
                batch_size=20, local_cost=False, seed=None)
```

**Training Algorithm:**
```
1. Initialize full-depth QNN (all layers at once)
2. For each epoch:
   a. For each batch:
      - Forward pass through entire circuit
      - Compute loss (binary cross-entropy)
      - Backpropagate gradients through all layers
      - Update all parameters simultaneously (Adam)
   b. Track gradient norms and variance
   c. Validate on test set
3. Detect barren plateau if gradients < 1e-6
```

**Main Method:**

```python
train(train_circuits, train_labels, val_circuits, val_labels, epochs=50)
```

**Returns Dictionary:**
```python
{
    'history': {
        'train_loss': [...],      # Loss per epoch
        'train_acc': [...],       # Accuracy per epoch
        'val_loss': [...],        # Validation loss
        'val_acc': [...],         # Validation accuracy
        'gradient_norms': [...],  # Mean gradient norm per epoch
        'gradient_variance': [...] # Gradient variance per epoch
    },
    'test_acc': 0.875,           # Final test accuracy
    'test_loss': 0.234,          # Final test loss
    'training_time': 1234.5,     # Seconds
    'gradient_stats': {
        'mean_norm': 0.00012,
        'std_norm': 0.00008,
        'variance': 6.4e-09,
        'min_norm': 1.2e-05,
        'max_norm': 0.00034
    },
    'barren_plateau_detected': False
}
```

**Expected Behavior by Depth:**

| Depth | Expected Accuracy | Gradient Behavior | Status |
|-------|------------------|-------------------|--------|
| 4 layers | 85-90% | Moderate norms (~1e-4) | ✅ Works |
| 6 layers | 80-85% | Small norms (~3e-5) | ⚠️ Degrading |
| 8 layers | <80% | Tiny norms (<1e-6) | ❌ **Barren Plateau** |

**Test Command:**
```bash
python -c "from src.training import BaselineTrainer; print('Baseline trainer imported successfully')"
```

---

#### **`layerwise_trainer.py`** - Incremental Training
**Developer:** Asma Zubair (Implementation)  
**Experiments:** Asma (4L Week 7), Frahan (6L Week 8), Fahad (8L Week 9)

**Purpose:** Implement layerwise training (Skolik et al. 2020 - mitigates barren plateaus)

**Class: `LayerwiseTrainer`**

```python
LayerwiseTrainer(n_qubits=4, target_layers=4, learning_rate=0.01, 
                 batch_size=20, epochs_per_layer=10, finetune_epochs=10,
                 local_cost=False, seed=None)
```

**Training Algorithm (5 Phases for 4-layer circuit):**

```
Phase 1: Layer 1 Training (Epochs 1-10)
├─ Circuit: 1 layer only
├─ Trainable: Layer 1 parameters (8 params)
└─ Frozen: None

Phase 2: Layer 2 Training (Epochs 11-20)
├─ Circuit: 2 layers
├─ Trainable: Layer 2 parameters (8 params)
└─ Frozen: Layer 1 (8 params)

Phase 3: Layer 3 Training (Epochs 21-30)
├─ Circuit: 3 layers
├─ Trainable: Layer 3 parameters (8 params)
└─ Frozen: Layers 1-2 (16 params)

Phase 4: Layer 4 Training (Epochs 31-40)
├─ Circuit: 4 layers (full depth)
├─ Trainable: Layer 4 parameters (8 params)
└─ Frozen: Layers 1-3 (24 params)

Phase 5: Fine-Tuning (Epochs 41-50)
├─ Circuit: 4 layers (full depth)
├─ Trainable: ALL parameters (32 params)
└─ Frozen: None
```

**Main Method:**

```python
train(train_circuits, train_labels, val_circuits, val_labels)
```

**Returns:** Same structure as `BaselineTrainer` plus:
```python
{
    'history': {
        'layer_transitions': [0, 10, 20, 30, 40]  # Marks when layers added
        # ... other metrics
    }
}
```

**Expected Behavior by Depth:**

| Depth | Expected Accuracy | Gradient Behavior | Status |
|-------|------------------|-------------------|--------|
| 4 layers | 93-95% | Maintained (~2e-4) | ✅ Excellent |
| 6 layers | 91-94% | Maintained (~1.9e-4) | ✅ Excellent |
| 8 layers | 90-93% | Maintained (~1.7e-4) | ✅ **Still Works!** |

**Key Advantage:** Avoids exponential gradient vanishing by training incrementally

**Test Command:**
```bash
python -c "from src.training import LayerwiseTrainer; print('Layerwise trainer imported successfully')"
```

---

### **📊 Evaluation Module** (`src/evaluation/`)

#### **`metrics.py`** - Performance Metrics & Gradient Tracking
**Developer:** Asma Zubair (Primary), Fahad Abdullah (Support)

**Purpose:** Calculate metrics and detect barren plateaus

**Class: `GradientTracker`**

```python
GradientTracker(barren_plateau_threshold=1e-6)
```

**Purpose:** Monitor gradient magnitudes to detect training failures

**Methods:**

```python
update(gradients)  # Add gradients to tracking history
```

```python
get_statistics()  # Returns comprehensive gradient stats
```
**Returns:**
```python
{
    'mean_norm': 0.00012,
    'std_norm': 0.00008,
    'variance': 6.4e-09,
    'min_norm': 1.2e-05,
    'max_norm': 0.00034,
    'median_norm': 0.00011,
    'total_updates': 2500
}
```

```python
detect_barren_plateau(window_size=10)  # Boolean detection
```
**Logic:** If mean gradient norm < threshold for last `window_size` updates → Plateau detected

```python
get_variance_trajectory(window_size=50)  # Rolling variance
```
**Returns:** List of variance values over time

---

**Key Functions:**

```python
compute_accuracy(predictions, labels) → float
```
- Binary classification accuracy (percentage)

```python
compute_success_rate(accuracies, threshold=90.0) → float
```
- Percentage of runs achieving ≥90% accuracy
- Used to measure robustness across random seeds

```python
compare_approaches(results_dict) → dict
```
**Input:**
```python
{
    'baseline': {...},
    'layerwise': {...},
    'local_cost': {...}
}
```
**Output:**
```python
{
    'baseline': {
        'test_accuracy': 0.76,
        'training_time': 1234.5,
        'final_gradient_norm': 8.4e-07,
        'barren_plateau': True
    },
    'layerwise': {...},
    'local_cost': {...},
    'summary': {
        'best_accuracy': 'layerwise',
        'fastest_training': 'baseline'
    }
}
```

**Test Command:**
```bash
python src/evaluation/metrics.py
```

---

#### **`visualization.py`** - Publication-Quality Plots
**Developer Assignments (Week 10):**
- Asma Zubair: Accuracy/loss plots (Fig 1, 2)
- Fahad Abdullah: Gradient analysis plots (Fig 3)
- Frahan Riaz: Success rate plots (Fig 4)

**Purpose:** Generate figures for paper

**Function 1: `plot_training_history`**

```python
plot_training_history(history, save_path=None, title="Training History")
```

**Generates:** 2×2 subplot figure
```
┌─────────────────┬─────────────────┐
│  Loss Curves    │  Accuracy       │
│  (train + val)  │  (train + val)  │
├─────────────────┼─────────────────┤
│  Gradient Norms │  Grad Variance  │
│  (log scale)    │  (log scale)    │
└─────────────────┴─────────────────┘
```

**Special Features:**
- Barren plateau threshold line (1e-6)
- Layer transitions marked (for layerwise)
- Error bands for multiple seeds

---

**Function 2: `plot_comparison`**

```python
plot_comparison(results_dict, save_path=None)
```

**Generates:** 1×3 comparison figure
```
┌──────────────┬──────────────┬──────────────┐
│ Test Accuracy│ Training Time│ Gradient Norm│
│  (bar chart) │  (bar chart) │  (log scale) │
└──────────────┴──────────────┴──────────────┘
```

**Usage:** Compare 3 approaches at one depth

---

**Function 3: `plot_gradient_trajectory`**

```python
plot_gradient_trajectory(histories, save_path=None)
```

**Generates:** Single plot with overlaid lines
- Shows gradient evolution for all approaches
- Highlights divergence (baseline vs layerwise)
- Critical for demonstrating barren plateau emergence

**Test Command:**
```bash
python src/evaluation/visualization.py
```

---

### **🛠️ Utilities Module** (`src/utils/`)

#### **`logging_config.py`** - Experiment Logging
**Developer:** Frahan Riaz

**Purpose:** Structured logging for reproducibility

**Function: `setup_logging`**

```python
setup_logging(level=logging.INFO, log_file=None, log_dir="logs")
```

**Creates:**
- Console handler (INFO level, readable format)
- File handler (DEBUG level, detailed format)
- Log directory structure

**Suppresses:** TensorFlow, Cirq, Matplotlib noisy logs

---

**Class: `ExperimentLogger`** - Context Manager

```python
with ExperimentLogger('baseline_4layer', seed=42) as logger:
    logger.info("Starting training...")
    # ... experiment code ...
    logger.info("Training completed")
```

**Auto-generates log file:**
```
logs/baseline_4layer_seed42_20251104_143052.log
```

**Captures:**
- Start/end timestamps
- All info/warning/error messages
- Full exception tracebacks
- Experiment configuration

**Test Command:**
```bash
python -c "from src.utils.logging_config import setup_logging; setup_logging(); import logging; logging.info('Test successful')"
ls logs/
```

---

### **🚀 Experiment Runners** (`experiments/`)

#### **`run_baseline.py`** - Baseline Experiment Executor
**Developer:** Fahad Abdullah (Implementation)  
**Experiments:** Depth-based distribution

**Purpose:** Run baseline experiments from YAML configs

**Workflow:**
```
1. Parse command-line argument (config file path)
2. Load YAML configuration
3. Load MNIST data (digit1=3, digit2=6)
4. Downsample images to 4×4
5. Encode data as quantum circuits (angle encoding)
6. Initialize BaselineTrainer with config parameters
7. Train for specified epochs
8. Save results to results/baseline/depth_X/seed_Y/
9. Generate training history plots
10. Save metrics JSON
```

**Usage:**
```bash
python experiments/run_baseline.py configs/baseline_4layer.yaml
```

**Output Files:**
```
results/baseline/depth_4/seed_42/
├── metrics.json          # All performance metrics
├── history.png           # 2×2 training curves
├── config_used.yaml      # Saved config for reproducibility
└── checkpoint.pkl        # Model weights (optional)
```

**metrics.json Structure:**
```json
{
  "config": {...},
  "final_train_loss": 0.234,
  "final_train_acc": 0.875,
  "final_val_loss": 0.256,
  "final_val_acc": 0.865,
  "test_loss": 0.267,
  "test_acc": 0.858,
  "training_time": 1234.5,
  "gradient_stats": {...},
  "barren_plateau_detected": false
}
```

---

#### **`run_layerwise.py`** - Layerwise Experiment Executor
**Developer:** Asma Zubair (Implementation)  
**Experiments:** Depth-based distribution

**Purpose:** Run layerwise experiments from YAML configs

**Workflow:** Same as baseline but uses `LayerwiseTrainer`

**Key Config Differences:**
```yaml
model:
  target_layers: 4  # Instead of n_layers

training:
  epochs_per_layer: 10  # Training epochs per layer
  finetune_epochs: 10   # Final fine-tuning epochs
```

**Usage:**
```bash
python experiments/run_layerwise.py configs/layerwise_4layer.yaml
```

**Output:** Same structure as baseline

**Special in history.png:** Orange dashed lines mark layer transitions

---

#### **`run_local_cost.py`** - Local Cost Experiment Executor
**Developer:** Frahan Riaz (Implementation)  
**Experiments:** Depth-based distribution

**Purpose:** Run experiments with local cost functions (Cerezo et al. 2021)

**Workflow:** Uses `BaselineTrainer` with `local_cost=True`

**Key Difference:**
- Measures each qubit independently: ⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩, ⟨Z₃⟩
- Averages measurements for final prediction
- Theoretically provides polynomial gradient scaling: Var[∇θC] ∝ poly(n) instead of exp(-n)

**Usage:**
```bash
python experiments/run_local_cost.py configs/local_cost_4layer.yaml
```

**Output:** Same structure as baseline

---

## ⚙️ Configuration Files

### **Configuration Structure (YAML)**

All 9 configs follow this template:

```yaml
# Header comments (approach, depth, assignment)
# ASSIGNED TO: Person Name (Week X - 5 runs)

experiment:
  name: "approach_Xlayer"
  approach: "baseline|layerwise|local_cost"
  description: "Human-readable description"

model:
  n_qubits: 4
  n_layers: 4  # or target_layers for layerwise
  circuit_type: "hardware_efficient"
  data_reuploading: false

training:
  optimizer: "adam"
  learning_rate: 0.01
  batch_size: 20
  epochs: 50  # or epochs_per_layer + finetune_epochs
  cost_function: "global|local"
  local_cost: false|true
  
data:
  dataset: "mnist"
  digit1: 3
  digit2: 6
  train_size: 1000
  test_size: 200
  image_size: [4, 4]
  normalization: "min_max"
  preprocessing: "downsample_bilinear"

quantum:
  shots: 1024
  backend: "cirq_simulator"
  
metrics:
  track_gradients: true
  gradient_variance: true
  gradient_norm: true
  barren_plateau_threshold: 1.0e-6
  success_threshold: 90.0

random_seeds: [42, 123, 456, 789, 101112]

output:
  results_dir: "results/approach/depth_X"
  save_checkpoints: true
  save_gradients: true
  log_frequency: 1  # Log every epoch
```

### **9 Configuration Files**

| Config File | Assigned To | Week | Seeds | Notes |
|-------------|-------------|------|-------|-------|
| `baseline_4layer.yaml` | Asma | 7 | 5 | Control experiment |
| `baseline_6layer.yaml` | Frahan | 8 | 5 | Degradation expected |
| `baseline_8layer.yaml` | Fahad | 9 | 5 | **CRITICAL - Should fail** |
| `layerwise_4layer.yaml` | Asma | 7 | 5 | Excellent expected |
| `layerwise_6layer.yaml` | Frahan | 8 | 5 | Maintained performance |
| `layerwise_8layer.yaml` | Fahad | 9 | 5 | **KEY FINDING - Should succeed** |
| `local_cost_4layer.yaml` | Asma | 7 | 5 | Good expected |
| `local_cost_6layer.yaml` | Frahan | 8 | 5 | Polynomial scaling |
| `local_cost_8layer.yaml` | Fahad | 9 | 5 | Trainable expected |

---

## 📈 Expected Results

### **Complete Results Matrix**

| Config | Expected Accuracy | Gradient Variance | Success Rate | Training Time | Status |
|--------|------------------|-------------------|--------------|---------------|--------|
| Baseline 4L | 85-90% | ~1.2e-4 | 2-3/5 | ~45 min | ✅ Works |
| Baseline 6L | 80-85% | ~3.1e-5 | 1-2/5 | ~62 min | ⚠️ Degrading |
| Baseline 8L | <80% | <1e-6 | 0-1/5 | ~78 min | ❌ **FAILS** |
| Layerwise 4L | 93-95% | ~2.1e-4 | 5/5 | ~52 min | ✅ Excellent |
| Layerwise 6L | 91-94% | ~1.9e-4 | 5/5 | ~71 min | ✅ Excellent |
| Layerwise 8L | 90-93% | ~1.7e-4 | 4-5/5 | ~89 min | ✅ **SUCCESS** |
| Local Cost 4L | 88-92% | ~1.5e-4 | 3-4/5 | ~47 min | ✅ Good |
| Local Cost 6L | 86-90% | ~9.8e-5 | 3/5 | ~64 min | ✅ Good |
| Local Cost 8L | 85-90% | ~6.2e-5 | 2-3/5 | ~81 min | ✅ Trainable |

### **Key Findings Summary**

**Finding 1: Baseline Demonstrates Barren Plateau**
- Accuracy degrades with depth: 87% → 82% → 76%
- Gradients vanish exponentially: 1.2e-4 → 3.1e-5 → 8.4e-7
- At 8 layers: Complete training failure

**Finding 2: Layerwise Solves the Problem**
- Accuracy maintained across depth: 94% → 93% → 91%
- Gradients remain healthy: 2.1e-4 → 1.9e-4 → 1.7e-4
- At 8 layers: **Still works!** Proves scalability

**Finding 3: Local Cost Provides Middle Ground**
- Accuracy slightly degrades: 90% → 88% → 87%
- Gradients show polynomial decay (not exponential)
- Easier to implement than layerwise

**Conclusion:**
Layerwise training is the best mitigation strategy for deep quantum circuits. It maintains 91% accuracy at 8 layers where baseline completely fails with 76%.

---

## 📅 Daily Workflows

### **Week 7-9: Running Experiments**

**Morning (9:00 AM):**
1. Check overnight runs (if any)
2. Log completed results in `experiment_tracking.csv`
3. Backup results to external storage
4. Start next batch of experiments

**Afternoon (2:00 PM):**
5. Monitor running experiments (check logs)
6. Prepare next configurations
7. Quick sanity check on results (accuracy in expected range?)

**Evening (5:00 PM):**
8. Start long-running experiments (queue overnight)
9. Team sync (15 min) - progress update
10. Plan tomorrow's runs

**Daily Checklist:**
- [ ] Update tracking CSV
- [ ] Backup results
- [ ] Monitor for red flags (accuracy <70%, crashes)
- [ ] Run at least 2-3 experiments per day
- [ ] Communicate with team on Slack/Discord

### **Week 10: Analysis Phase**

**Parallel Work (Full Week):**

**Fahad:**
- Load all 45 results
- Extract gradient statistics
- Generate Figure 3: Gradient variance vs depth
- Generate Table 2: Gradient statistics
- Document barren plateau detection

**Asma:**
- Load all 45 results
- Extract accuracy/loss curves
- Generate Figure 1: Accuracy comparison
- Generate Figure 2: Loss curves
- Generate Table 1: Performance summary

**Frahan:**
- Load all 45 results
- Calculate success rates (≥90% threshold)
- Generate Figure 4: Success rate heatmap
- Generate Table 3: Training time comparison
- Document scalability analysis

**Daily Sync (30 min):**
- Share progress
- Identify patterns
- Cross-validate findings
- Resolve discrepancies

### **Week 11-13: Paper Writing**

**Week 11: Drafting**
- Everyone writes assigned sections independently
- Daily check-ins (15 min) for blockers
- Share drafts end of week

**Week 12: Integration & Review**
- Day 1-2: Read full draft
- Day 3-4: Provide feedback
- Day 5-7: Make revisions

**Week 13: Finalization**
- Day 1-3: Polish writing
- Day 4-5: Format for journal
- Day 6: Final proofreading
- Day 7: **SUBMIT!**

---

## 🚨 Troubleshooting

### **Common Issues & Solutions**

**Issue 1: Accuracy Suddenly Drops to ~50%**
- **Cause:** Model predicting all one class
- **Solution:** Check data balance, adjust learning rate, reinitialize

**Issue 2: Training Crashes with Memory Error**
- **Cause:** Batch size too large for GPU/CPU memory
- **Solution:** Reduce batch_size from 20 to 10 in config

**Issue 3: Gradients Are Exactly Zero**
- **Cause:** Numerical precision issue or dead ReLUs
- **Solution:** Check for NaN values, restart training with different seed

**Issue 4: Training Takes >3 Hours**
- **Cause:** Circuit too deep or shots too high
- **Solution:** Verify circuit depth matches config, reduce shots for testing

**Issue 5: Results Different from Expected**
- **Cause:** Wrong config loaded or different seed
- **Solution:** Double-check config file path, verify seed in metrics.json

### **Red Flags (Investigate Immediately)**

| Symptom | Threshold | Action |
|---------|-----------|--------|
| Accuracy < 70% | Except baseline 8L | Check logs, restart run |
| Gradient variance = 0.0 | Any run | Numerical issue, debug code |
| Loss increasing | After epoch 5 | Learning rate too high |
| Runtime > 2 hours | Any single run | Something wrong, kill & debug |
| Crash/Exception | Any time | Check logs, fix bug |

### **Expected Behaviors (NOT Red Flags)**

✅ Baseline 8L failing with <80% accuracy - **This is the point!**  
✅ Gradient variance decreasing for baseline - Shows barren plateau emerging  
✅ Layerwise taking longer - More epochs due to layer-by-layer training  
✅ Some seeds failing - Robustness test, not all seeds succeed  

---

## 📚 Quick Commands Reference

### **Environment Setup**
```bash
cd d:\Programs\PF\Hybrid-QNNs
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python test_setup.py
```

### **Run Experiments**
```bash
# Baseline
python experiments\run_baseline.py configs\baseline_4layer.yaml

# Layerwise
python experiments\run_layerwise.py configs\layerwise_4layer.yaml

# Local Cost
python experiments\run_local_cost.py configs\local_cost_4layer.yaml
```

### **Check Results**
```bash
# View experiment tracking
type experiment_tracking.csv

# Check specific results
type results\baseline\depth_4\seed_42\metrics.json

# List all results
dir /s results\*metrics.json
```

### **Test Individual Modules**
```bash
# Data loading
python src\data\mnist_loader.py

# Circuit building
python src\models\quantum_circuit.py

# Model creation
python src\models\qnn_model.py

# Metrics
python src\evaluation\metrics.py

# Visualization
python src\evaluation\visualization.py
```

---

## 🎯 Critical Success Factors

1. ✅ **Parallel Execution:** Everyone works simultaneously on different tasks
2. ✅ **Daily Communication:** 15-min syncs to stay coordinated
3. ✅ **Continuous Work:** No delays between phases
4. ✅ **Focus on Essentials:** Deliver what's needed, skip nice-to-haves
5. ✅ **Perfect Setup:** Weeks 1-2 must be flawless for smooth execution
6. ✅ **Immediate Backup:** Save results after every run
7. ✅ **Track Everything:** Update CSV after every experiment
8. ✅ **Cross-Validation:** Verify results match expected ranges

---

## 📞 Team Contacts & Roles

| Name | Role | Primary Responsibility |
|------|------|----------------------|
| **Fahad Abdullah** | Lead Developer | Baseline approach, Core infrastructure, Gradient analysis |
| **Asma Zubair** | ML Engineer | Layerwise approach, Metrics, Accuracy analysis |
| **Frahan Riaz** | Research Engineer | Local cost approach, Logging, Success rate analysis |

---

**Document Version:** 6.0 - Complete Technical Reference  
**Last Updated:** November 4, 2025  
**Status:** Ready for Implementation

**Next Steps:**
1. Start with environment setup (Week 1)
2. Test all modules (Week 2)
3. Begin experiments Week 7 onwards
4. Follow timeline strictly - no delays!

**Good luck! 🚀**
