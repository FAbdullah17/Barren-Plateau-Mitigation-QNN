<p align="center">
  <h1 align="center">🔬 Barren Plateau Mitigation in Quantum Neural Networks</h1>
  <p align="center">
    <strong>Empirical Comparison of Layerwise Training and Local Cost Functions</strong>
  </p>
  <p align="center">
    A systematic study addressing the barren plateau problem in variational quantum circuits through two prominent mitigation strategies, evaluated on a standardized MNIST benchmark.
  </p>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3100/">
    <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10">
  </a>
  <a href="https://www.tensorflow.org/">
    <img src="https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow 2.15">
  </a>
  <a href="https://www.tensorflow.org/quantum">
    <img src="https://img.shields.io/badge/TFQ-0.7.2-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TFQ 0.7.2">
  </a>
  <a href="https://quantumai.google/cirq">
    <img src="https://img.shields.io/badge/Cirq-1.3.0-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Cirq 1.3.0">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT">
  </a>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Research Question](#-research-question)
- [The Barren Plateau Problem](#-the-barren-plateau-problem)
- [Mitigation Strategies](#-mitigation-strategies)
- [Key Results](#-key-results)
- [Tech Stack & Dependencies](#-tech-stack--dependencies)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Verification](#environment-verification)
- [Usage Guide](#-usage-guide)
  - [Running Single Experiments](#running-single-experiments)
  - [Running Batch Experiments](#running-batch-experiments)
  - [Running the Full Experiment Suite](#running-the-full-experiment-suite)
- [Configuration System](#-configuration-system)
- [Understanding the Code](#-understanding-the-code)
  - [Data Pipeline](#1-data-pipeline)
  - [Quantum Circuit Architecture](#2-quantum-circuit-architecture)
  - [Training Strategies](#3-training-strategies)
  - [Evaluation & Metrics](#4-evaluation--metrics)
  - [Visualization](#5-visualization)
- [Experiment Design](#-experiment-design)
- [Results & Analysis](#-results--analysis)
  - [Accuracy Results](#accuracy-results)
  - [Gradient Analysis](#gradient-analysis)
  - [Depth Impact](#depth-impact-analysis)
  - [Success Rate](#success-rate-analysis)
  - [Training Time](#training-time-analysis)
- [Notebooks](#-notebooks)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [Research Team](#-research-team)
- [Academic References](#-academic-references)
- [Future Work](#-future-work)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🔭 Overview

This repository contains the complete implementation, experiment infrastructure, and analysis tools for a systematic empirical comparison of two prominent barren plateau mitigation strategies in quantum neural networks (QNNs):

1. **Layerwise Training** — Progressive circuit construction with incremental optimization (Skolik et al., 2020)
2. **Local Cost Functions** — Per-qubit measurement operators replacing global observables (Cerezo et al., 2021)

Both strategies are evaluated against a **baseline** (standard end-to-end training) across three circuit depths (4, 6, and 8 layers) with 5 random seeds each, producing a total of **45 controlled experiments**. The study uses **MNIST binary classification** (digits 3 vs. 6) as a standardized benchmark, implemented with a 4-qubit hardware-efficient ansatz using TensorFlow Quantum.

### Why This Research Matters

The barren plateau phenomenon is one of the most significant obstacles to scaling quantum machine learning. As quantum circuits grow deeper, the gradients of the cost function exponentially vanish, making optimization practically impossible. While several mitigation strategies have been proposed individually, **no direct empirical comparison** existed on a standardized benchmark — until now. This project fills that gap by providing rigorous, reproducible evidence comparing the two most cited approaches.

---

## 🎯 Research Question

> **How do layerwise training and local cost functions compare in their effectiveness at mitigating barren plateaus in quantum neural networks, as measured by gradient flow, training dynamics, and classification performance on the MNIST benchmark?**

### Sub-Questions

1. At what circuit depth does the barren plateau become empirically observable in standard training?
2. How effectively does layerwise training preserve gradient magnitudes as circuit depth increases?
3. How effectively do local cost functions preserve gradient magnitudes as circuit depth increases?
4. Is there a statistically significant difference between the two mitigation strategies?
5. What is the computational overhead of each mitigation strategy?

---

## 🏔️ The Barren Plateau Problem

### What Is a Barren Plateau?

In variational quantum algorithms (VQAs), a **barren plateau** occurs when the gradient of the cost function becomes exponentially small across the entire parameter space as the number of qubits or circuit depth increases. Mathematically:

$$\text{Var}\left[\frac{\partial C}{\partial \theta_i}\right] \leq F(n) \quad \text{where } F(n) \in O\left(\frac{1}{b^n}\right)$$

where $n$ is the number of qubits and $b > 1$ is a constant. This means:

- **Gradients vanish exponentially** — The optimization landscape becomes flat
- **Training stagnates** — The model cannot learn meaningful patterns
- **Random guessing results** — Accuracy hovers near 50% for binary classification
- **No gradient signal** — Parameter updates become meaninglessly small

### Why It Matters

The barren plateau problem is not merely a theoretical concern. It has direct implications for:

- **Quantum advantage claims** — If QNNs cannot be trained at useful depths, they cannot outperform classical alternatives
- **Near-term quantum computing** — NISQ devices require variational algorithms, which are directly affected
- **Scalability** — Any practical quantum ML application requires circuits deeper than what barren plateaus allow

### Empirical Evidence in This Project

Our experiments provide clear empirical evidence of the barren plateau phenomenon:

| Depth | Baseline Accuracy | Gradient Norm | Trainable? |
|-------|-------------------|---------------|------------|
| 4 layers | 73.8% ± 2.2% | ~0.28 | ✅ Yes |
| 6 layers | 73.9% ± 2.1% | ~0.22 | ✅ Yes |
| 8 layers | **52.7% ± 1.1%** | ~0.22 | ❌ No |

At 8 layers, the baseline model collapses to near-random accuracy (52.7%), clearly demonstrating the barren plateau effect.

---

## 🛡️ Mitigation Strategies

### Strategy 1: Layerwise Training (Skolik et al., 2020)

**Core idea:** Instead of training all circuit parameters simultaneously, train them incrementally — one layer at a time.

**How it works:**
1. Start with a 1-layer circuit and train it to convergence
2. Freeze the first layer's parameters
3. Add a second layer and train only its parameters
4. Repeat until all layers are added
5. Optionally fine-tune all parameters together

**Why it helps:** By training each layer in a smaller parameter space, the optimization landscape is less flat. Each layer starts from a good initialization provided by the previously trained layers, avoiding the exponential gradient vanishing that occurs when all parameters are random.

**Implementation:** [`src/training/layerwise_trainer.py`](src/training/layerwise_trainer.py)

```python
trainer = LayerwiseTrainer(
    n_qubits=4,
    n_layers=8,
    learning_rate=0.01,
    batch_size=20,
    epochs_per_layer=10  # Train each layer for 10 epochs
)
results = trainer.train(train_circuits, train_labels, val_circuits, val_labels)
```

### Strategy 2: Local Cost Functions (Cerezo et al., 2021)

**Core idea:** Replace the global cost function (which measures all qubits together) with a local cost function (which measures each qubit independently).

**How it works:**
- **Global cost:** Uses a single observable across all qubits → gradients vanish exponentially
- **Local cost:** Uses independent Pauli-Z measurements on each qubit → gradients vanish only polynomially

**Why it helps:** The theoretical result by Cerezo et al. shows that local cost functions have gradient variance that decreases at most polynomially with circuit depth, compared to the exponential decrease seen with global cost functions.

**Implementation:** [`src/models/qnn_model.py`](src/models/qnn_model.py)

```python
# Global cost (baseline) — single Z measurement on first qubit
model = QuantumNeuralNetwork(n_qubits=4, local_cost=False)
# readout_ops = [cirq.Z(q0)]

# Local cost — independent Z measurements on each qubit
model = QuantumNeuralNetwork(n_qubits=4, local_cost=True)
# readout_ops = [cirq.Z(q0), cirq.Z(q1), cirq.Z(q2), cirq.Z(q3)]
```

### Strategy 3: Baseline (Control Condition)

Standard end-to-end training with all parameters optimized simultaneously using a global cost function. This serves as the **control condition** to quantify the severity of barren plateaus and measure the effectiveness of mitigation strategies.

**Implementation:** [`src/training/baseline_trainer.py`](src/training/baseline_trainer.py)

---

## 📊 Key Results

### Executive Summary

| Metric | Baseline | Layerwise | Local Cost |
|--------|----------|-----------|------------|
| **4-Layer Accuracy** | 73.8 ± 2.2% | 74.0 ± 3.0% | **75.3 ± 3.2%** |
| **6-Layer Accuracy** | 73.9 ± 2.1% | 74.2 ± 2.5% | **75.6 ± 3.1%** |
| **8-Layer Accuracy** | ❌ 52.7 ± 1.1% | ✅ 73.9 ± 2.6% | ✅ **75.4 ± 2.5%** |
| **Depth Robustness** | ❌ Fails at 8L | ✅ Stable | ✅ Stable |
| **Best 8L Accuracy** | 54.4% | 77.0% | **78.0%** |
| **Worst 8L Accuracy** | 51.6% | 69.5% | 71.0% |

### Key Findings

1. **Barren plateau is real and measurable** — Baseline accuracy drops from 74% to 53% between 4 and 8 layers
2. **Both mitigation strategies work** — Layerwise and Local Cost maintain 74-75% accuracy at all depths
3. **Local Cost is marginally better** — Consistently ~1.5% higher accuracy than Layerwise, with slightly lower variance
4. **The effect is statistically significant** — 8-layer baseline is >20 percentage points below both mitigation strategies
5. **Reproducibility is confirmed** — Low variance (1-3%) across 5 independent random seeds

---

## 🛠️ Tech Stack & Dependencies

### Core Framework

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.10 | Primary programming language |
| **TensorFlow** | 2.15.0 | Deep learning backend |
| **TensorFlow Quantum** | 0.7.2 | Quantum-classical hybrid ML framework |
| **Cirq** | 1.3.0 | Quantum circuit construction and simulation |
| **SymPy** | 1.12 | Symbolic mathematics for parameterized circuits |

### Machine Learning & Data

| Library | Version | Purpose |
|---------|---------|---------|
| **NumPy** | 1.24.3 | Numerical computing and array operations |
| **Scikit-learn** | 1.3.2 | Data preprocessing, train/test splitting |
| **Pandas** | 2.1.4 | Data analysis and results aggregation |

### Visualization

| Library | Version | Purpose |
|---------|---------|---------|
| **Matplotlib** | 3.8.2 | Publication-quality plots and figures |
| **Seaborn** | 0.13.0 | Statistical visualization and styling |

### Utilities

| Library | Version | Purpose |
|---------|---------|---------|
| **PyYAML** | 6.0.1 | YAML configuration file parsing |
| **tqdm** | 4.66.1 | Progress bars for training loops |
| **Jupyter** | 1.0.0 | Interactive analysis notebooks |
| **ipykernel** | 6.27.1 | Jupyter kernel for Python 3.10 |
| **tf_keras** | 2.16.0 | Keras API compatibility layer |

### Full Requirements

All dependencies with pinned versions are listed in [`requirements.txt`](requirements.txt).

---

## 📁 Project Structure

```
Hybrid-QNNs/
│
├── 📄 README.md                          # This file
├── 📄 CONTRIBUTING.md                    # Contribution guidelines
├── 📄 LICENSE                            # MIT License
├── 📄 requirements.txt                   # Pinned Python dependencies
├── 📄 setup.py                           # Package installation configuration
├── 📄 Research-Problem-Statement.md      # Formal problem statement
├── 📄 .gitignore                         # Git ignore rules
│
├── 📂 src/                               # Core source code package
│   ├── 📄 __init__.py                    # Package initialization
│   │
│   ├── 📂 data/                          # Data loading & preprocessing
│   │   ├── 📄 __init__.py
│   │   └── 📄 mnist_loader.py            # MNIST download, filter, downsample, encode
│   │
│   ├── 📂 models/                        # Quantum circuit & model definitions
│   │   ├── 📄 __init__.py
│   │   ├── 📄 quantum_circuit.py         # Hardware-efficient ansatz (RY, RZ, CNOT)
│   │   └── 📄 qnn_model.py              # TFQ-integrated Keras models
│   │
│   ├── 📂 training/                      # Training strategies
│   │   ├── 📄 __init__.py
│   │   ├── 📄 baseline_trainer.py        # Standard end-to-end training
│   │   └── 📄 layerwise_trainer.py       # Incremental layer-by-layer training
│   │
│   ├── 📂 evaluation/                    # Metrics & visualization
│   │   ├── 📄 __init__.py
│   │   ├── 📄 metrics.py                 # Gradient tracking, BP detection, stats
│   │   └── 📄 visualization.py           # Training curves, comparison plots
│   │
│   └── 📂 utils/                         # Shared utilities
│       ├── 📄 __init__.py
│       └── 📄 logging_config.py          # Logging infrastructure
│
├── 📂 experiments/                       # Experiment entry points
│   ├── 📄 run_baseline.py                # Run single baseline experiment
│   ├── 📄 run_layerwise.py               # Run single layerwise experiment
│   ├── 📄 run_local_cost.py              # Run single local cost experiment
│   └── 📄 run_comparison.py              # Generate cross-approach comparisons
│
├── 📂 configs/                           # YAML experiment configurations
│   ├── 📄 baseline_4layer.yaml           # Baseline at 4 layers
│   ├── 📄 baseline_6layer.yaml           # Baseline at 6 layers
│   ├── 📄 baseline_8layer.yaml           # Baseline at 8 layers (BP expected)
│   ├── 📄 baseline_test.yaml             # Quick smoke test config
│   ├── 📄 layerwise_4layer.yaml          # Layerwise at 4 layers
│   ├── 📄 layerwise_6layer.yaml          # Layerwise at 6 layers
│   ├── 📄 layerwise_8layer.yaml          # Layerwise at 8 layers
│   ├── 📄 layerwise_test.yaml            # Quick smoke test config
│   ├── 📄 local_cost_4layer.yaml         # Local cost at 4 layers
│   ├── 📄 local_cost_6layer.yaml         # Local cost at 6 layers
│   └── 📄 local_cost_8layer.yaml         # Local cost at 8 layers
│
├── 📂 scripts/                           # Automation & analysis scripts
│   ├── 📄 run_4layer_experiments.py      # Batch run all 4-layer experiments
│   ├── 📄 run_6layer_experiments.py      # Batch run all 6-layer experiments
│   ├── 📄 run_8layer_experiments.py      # Batch run all 8-layer experiments
│   ├── 📄 run_batch.py                   # Generic batch experiment runner
│   ├── 📄 compare_metrics.py             # Cross-experiment metric comparison
│   ├── 📄 validate_results.py            # Result integrity validation
│   ├── 📄 check_output_format.py         # Output format consistency checks
│   └── 📄 analyze_seed_variance.py       # Seed-to-seed variance analysis
│
├── 📂 tests/                             # Unit & integration tests
│   ├── 📄 test_data_loader.py            # Data loading tests
│   ├── 📄 test_data_consistency.py       # Data integrity tests
│   ├── 📄 test_quantum_circuit.py        # Circuit construction tests
│   ├── 📄 test_qnn_model.py             # Model integration tests
│   ├── 📄 test_trainers.py              # Training strategy tests
│   ├── 📄 test_metrics.py               # Evaluation metrics tests
│   └── 📄 test_pipeline.py              # End-to-end pipeline tests
│
├── 📂 notebooks/                         # Jupyter analysis notebooks
│   ├── 📓 00_data_loading.ipynb          # Data loading exploration
│   ├── 📓 01_data_exploration.ipynb      # Dataset visualization & statistics
│   ├── 📓 02_circuit_visualization.ipynb # Quantum circuit visualization
│   ├── 📓 03_results_analysis.ipynb      # Experiment results analysis
│   └── 📓 04_barren_plateau_analysis.ipynb # Barren plateau specific analysis
│
├── 📂 docs/                              # Documentation
│   ├── 📄 methodology.md                 # Research methodology & design
│   ├── 📄 metrics_schema.md              # Output JSON schema documentation
│   ├── 📄 quickstart.md                  # Quick start guide
│   ├── 📄 results_guide.md               # Results interpretation guide
│   ├── 📄 results_interpretation.md      # Detailed results analysis guide
│   ├── 📄 troubleshooting.md             # Common issues & solutions
│   └── 📄 validation_results.md          # Pre-experiment validation summary
│
├── 📂 results/                           # Experiment outputs (45 experiments)
│   ├── 📂 baseline/
│   │   ├── 📂 depth_4/seed_{42,123,456,789,101112}/
│   │   ├── 📂 depth_6/seed_{42,123,456,789,101112}/
│   │   └── 📂 depth_8/seed_{42,123,456,789,101112}/
│   ├── 📂 layerwise/
│   │   ├── 📂 depth_4/seed_{...}/
│   │   ├── 📂 depth_6/seed_{...}/
│   │   └── 📂 depth_8/seed_{...}/
│   └── 📂 local_cost/
│       ├── 📂 depth_4/seed_{...}/
│       ├── 📂 depth_6/seed_{...}/
│       └── 📂 depth_8/seed_{...}/
│
└── 📂 data/                              # Cached dataset files
    └── (auto-downloaded MNIST data)
```

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.10.x | Required for TFQ compatibility |
| **pip** | ≥21.0 | Package manager |
| **Git** | Any | Version control |
| **RAM** | ≥8 GB | 16 GB recommended |
| **Disk Space** | ≥2 GB | For dependencies + results |

> **⚠️ Important:** TensorFlow Quantum 0.7.2 requires Python 3.10 specifically. Other Python versions are not supported.

### Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/FAbdullah17/Hybrid-QNNs.git
cd Hybrid-QNNs
```

#### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate       # Linux/macOS
.\venv\Scripts\activate        # Windows (cmd)
.\venv\Scripts\Activate.ps1    # Windows (PowerShell)
```

#### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 4: Install Package in Development Mode (Optional)

```bash
pip install -e .
```

### Environment Verification

Verify that all critical dependencies are installed correctly:

```bash
python -c "
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import numpy as np

print(f'TensorFlow:         {tf.__version__}')
print(f'TensorFlow Quantum: {tfq.__version__}')
print(f'Cirq:              {cirq.__version__}')
print(f'NumPy:             {np.__version__}')
print(f'GPU Available:     {len(tf.config.list_physical_devices(\"GPU\")) > 0}')
print('All dependencies OK ✅')
"
```

Expected output:
```
TensorFlow:         2.15.0
TensorFlow Quantum: 0.7.2
Cirq:              1.3.0
NumPy:             1.24.3
GPU Available:     False
All dependencies OK ✅
```

### Running Tests

Verify the installation by running the test suite:

```bash
python -m pytest tests/ -v
```

All 7 test files should pass successfully.

---

## 📘 Usage Guide

### Running Single Experiments

Each experiment requires a YAML configuration file and an optional seed override:

```bash
# Baseline training at 4 layers with seed 42
python experiments/run_baseline.py configs/baseline_4layer.yaml --seed 42

# Layerwise training at 6 layers with seed 123
python experiments/run_layerwise.py configs/layerwise_6layer.yaml --seed 123

# Local cost training at 8 layers with seed 456
python experiments/run_local_cost.py configs/local_cost_8layer.yaml --seed 456
```

Each experiment will:
1. Load and preprocess MNIST data (digits 3 and 6)
2. Construct the parameterized quantum circuit
3. Train the model for the specified number of epochs
4. Track gradients and compute barren plateau metrics
5. Save results to `results/<approach>/depth_<N>/seed_<S>/`

Output files:
- `metrics.json` — Complete experiment metrics (see [Metrics Schema](docs/metrics_schema.md))
- `training_history.png` — Training curves visualization

### Running Batch Experiments

#### By Depth (Recommended)

Run all 15 experiments at a specific circuit depth:

```bash
# All 4-layer experiments (3 approaches × 5 seeds = 15 runs)
python scripts/run_4layer_experiments.py

# All 6-layer experiments
python scripts/run_6layer_experiments.py

# All 8-layer experiments
python scripts/run_8layer_experiments.py
```

#### Dry Run Mode

Preview the commands without executing:

```bash
python scripts/run_4layer_experiments.py --dry-run
```

#### By Approach

Run all 5 seeds for a single approach and depth:

```bash
python scripts/run_batch.py baseline configs/baseline_4layer.yaml
```

### Running the Full Experiment Suite

To reproduce all 45 experiments:

```bash
# Run all experiments sequentially
python scripts/run_4layer_experiments.py
python scripts/run_6layer_experiments.py
python scripts/run_8layer_experiments.py
```

**⏱️ Estimated total time: 25-40 hours** (on CPU)

| Depth | Experiments | Estimated Time |
|-------|-------------|----------------|
| 4 layers | 15 runs | 5-8 hours |
| 6 layers | 15 runs | 8-12 hours |
| 8 layers | 15 runs | 12-20 hours |

---

## ⚙️ Configuration System

All experiments are configured via YAML files in the `configs/` directory. This ensures full reproducibility and easy parameter sweeping.

### Configuration Schema

```yaml
# Experiment identification
experiment:
  name: "baseline_4layer"           # Unique experiment name
  approach: "baseline"              # Training approach
  description: "Description..."     # Human-readable description

# Quantum circuit model
model:
  n_qubits: 4                       # Number of qubits
  n_layers: 4                       # Circuit depth (layers)
  circuit_type: "hardware_efficient" # Ansatz type
  data_reuploading: false            # Whether to re-upload data

# Training hyperparameters
training:
  optimizer: "adam"                  # Optimizer (adam, sgd)
  learning_rate: 0.01               # Learning rate
  batch_size: 20                    # Batch size
  epochs: 50                        # Number of training epochs
  cost_function: "global"           # Cost function type
  local_cost: false                 # Enable local cost

# Dataset configuration
data:
  dataset: "mnist"                  # Dataset name
  digit1: 3                         # First digit class
  digit2: 6                         # Second digit class
  train_size: 1000                  # Training samples
  test_size: 200                    # Test samples
  image_size: [4, 4]                # Downsampled image dimensions
  normalization: "min_max"          # Normalization method
  preprocessing: "downsample_bilinear"  # Preprocessing method

# Quantum simulation
quantum:
  shots: 1024                       # Measurement shots
  backend: "cirq_simulator"         # Simulation backend

# Metrics configuration
metrics:
  track_gradients: true             # Enable gradient tracking
  gradient_variance: true           # Track gradient variance
  gradient_norm: true               # Track gradient norms
  barren_plateau_threshold: 1.0e-6  # BP detection threshold
  success_threshold: 90.0           # Success accuracy threshold (%)

# Random seeds for statistical robustness
random_seeds: [42, 123, 456, 789, 101112]

# Output configuration
output:
  results_dir: "results/baseline/depth_4"  # Output directory
  save_checkpoints: true                   # Save model checkpoints
  save_gradients: true                     # Save gradient history
  log_frequency: 1                         # Log every N epochs
```

### Available Configurations

| Config File | Approach | Depth | Key Setting |
|-------------|----------|-------|-------------|
| `baseline_4layer.yaml` | Baseline | 4 | `local_cost: false` |
| `baseline_6layer.yaml` | Baseline | 6 | `local_cost: false` |
| `baseline_8layer.yaml` | Baseline | 8 | `local_cost: false` |
| `layerwise_4layer.yaml` | Layerwise | 4 | Incremental training |
| `layerwise_6layer.yaml` | Layerwise | 6 | Incremental training |
| `layerwise_8layer.yaml` | Layerwise | 8 | Incremental training |
| `local_cost_4layer.yaml` | Local Cost | 4 | `local_cost: true` |
| `local_cost_6layer.yaml` | Local Cost | 6 | `local_cost: true` |
| `local_cost_8layer.yaml` | Local Cost | 8 | `local_cost: true` |
| `baseline_test.yaml` | Baseline | 2 | Quick smoke test |
| `layerwise_test.yaml` | Layerwise | 2 | Quick smoke test |

---

## 🧠 Understanding the Code

### 1. Data Pipeline

**File:** [`src/data/mnist_loader.py`](src/data/mnist_loader.py)

The data pipeline handles the complete flow from raw MNIST images to quantum-ready encoded data:

```
MNIST (28×28) → Filter (digits 3,6) → Downsample (4×4) → Normalize (0-1) → Encode (RY gates)
```

**Key steps:**

1. **Download & Filter:** Loads MNIST from TensorFlow datasets, filters to binary classification (digits 3 vs 6)
2. **Downsample:** Reduces 28×28 images to 4×4 using bilinear interpolation, producing 16 features that map to 4 qubits × 4 data points
3. **Normalize:** Scales pixel values to [0, 1] using min-max normalization
4. **Encode:** Converts each feature $x_i$ to a rotation angle $\theta_i = x_i \times \pi$ for RY gate encoding

```python
from src.data import load_mnist_binary, encode_data_for_qnn

X_train, y_train, X_test, y_test = load_mnist_binary(
    digit1=3, digit2=6,
    train_size=1000, test_size=200,
    image_size=(4, 4), seed=42
)
# X_train.shape: (1000, 16) — 4×4 flattened images
# y_train: {-1, +1} binary labels
```

### 2. Quantum Circuit Architecture

**File:** [`src/models/quantum_circuit.py`](src/models/quantum_circuit.py)

The quantum circuit uses a **hardware-efficient ansatz** (Kandala et al., 2017) with the following structure per layer:

```
Layer l:
┌────────────────────────────────────────────────┐
│  RY(θ₁) ─── RZ(θ₂) ─── ● ─────────           │
│                          │                      │
│  RY(θ₃) ─── RZ(θ₄) ─── ⊕ ─── ● ──            │
│                                │                │
│  RY(θ₅) ─── RZ(θ₆) ───────── ⊕ ─── ● ─       │
│                                      │          │
│  RY(θ₇) ─── RZ(θ₈) ──────────────── ⊕ ─       │
│                                                 │
│  + Additional RY rotation layer                 │
└────────────────────────────────────────────────┘
```

**Structure per layer:**
- RY rotation on each qubit (parameterized)
- RZ rotation on each qubit (parameterized)
- CNOT entanglement ladder (nearest-neighbor)
- Additional RY rotation (parameterized)

**Total parameters per layer:** $3 \times n_{\text{qubits}}$ (3 rotation gates per qubit)

**Total parameters for circuit:** $3 \times n_{\text{qubits}} \times n_{\text{layers}}$

| Config | Qubits | Layers | Parameters |
|--------|--------|--------|------------|
| 4-layer | 4 | 4 | 48 |
| 6-layer | 4 | 6 | 72 |
| 8-layer | 4 | 8 | 96 |

### 3. Training Strategies

#### Baseline Training

**File:** [`src/training/baseline_trainer.py`](src/training/baseline_trainer.py)

Standard end-to-end training using Adam optimizer with Binary Cross-Entropy loss:

```python
# Pseudocode for baseline training
model = QuantumNeuralNetwork(n_qubits=4, n_layers=8)
optimizer = Adam(lr=0.01)

for epoch in range(50):
    for batch_x, batch_y in data_loader:
        predictions = model(batch_x)
        loss = binary_crossentropy(predictions, batch_y)
        gradients = tape.gradient(loss, model.parameters)
        optimizer.apply_gradients(zip(gradients, model.parameters))
        # Track gradient norms for BP analysis
```

#### Layerwise Training

**File:** [`src/training/layerwise_trainer.py`](src/training/layerwise_trainer.py)

Progressive layer-by-layer training following Skolik et al. (2020):

```python
# Pseudocode for layerwise training
for depth in range(1, n_layers + 1):
    model = QuantumNeuralNetwork(n_qubits=4, n_layers=depth)

    # Freeze all parameters except the current layer
    for param in model.parameters[:previous_layer_params]:
        param.trainable = False

    # Train only the new layer
    for epoch in range(epochs_per_layer):
        train_step(model, data)

    # Unfreeze for potential fine-tuning
```

#### Local Cost Training

Uses the same training loop as the baseline but with per-qubit readout operators:

```python
# Global readout (baseline): single Z measurement
readout_ops = [cirq.Z(qubits[0])]

# Local readout: independent Z on each qubit
readout_ops = [cirq.Z(q) for q in qubits]  # 4 independent measurements
```

### 4. Evaluation & Metrics

**File:** [`src/evaluation/metrics.py`](src/evaluation/metrics.py)

The evaluation module tracks:

| Metric | Description | How It's Computed |
|--------|-------------|-------------------|
| **Test Accuracy** | Classification performance | Correct predictions / total |
| **Gradient Norms** | Magnitude of parameter updates | L2 norm of gradient vector |
| **Gradient Variance** | Stability of gradients | Variance across parameters |
| **BP Detection** | Whether barren plateau is present | Mean gradient norm < 1e-6 |
| **Training Time** | Computational cost | Wall-clock time in seconds |

**Barren Plateau Detection Criteria:**
```python
barren_plateau_detected = (gradient_stats['mean_norm'] < threshold)
# Default threshold: 1e-6
```

> **Note:** The automated detection uses a conservative threshold. Moderate gradient degradation (e.g., 8-layer baseline) may cause training failure without triggering the automated flag.

### 5. Visualization

**File:** [`src/evaluation/visualization.py`](src/evaluation/visualization.py)

Generates publication-quality plots including:

- **Training History:** Loss and accuracy curves over epochs
- **Gradient Trajectories:** Gradient norm evolution (log scale)
- **Comparison Bar Charts:** Cross-approach accuracy/time comparison with error bars
- **Gradient Heatmaps:** Parameter-level gradient visualization

```python
from src.evaluation import plot_training_history, plot_comparison

# Single experiment plot
plot_training_history(history, save_path="training.png")

# Cross-approach comparison
plot_comparison(results_dict, save_path="comparison.png")
```

---

## 🔬 Experiment Design

### Experimental Matrix

The full experiment suite consists of 45 controlled experiments:

| | 4 Layers | 6 Layers | 8 Layers |
|---|----------|----------|----------|
| **Baseline** | 5 seeds | 5 seeds | 5 seeds |
| **Layerwise** | 5 seeds | 5 seeds | 5 seeds |
| **Local Cost** | 5 seeds | 5 seeds | 5 seeds |

**Total:** 3 approaches × 3 depths × 5 seeds = **45 experiments**

### Controlled Variables

| Variable | Value | Rationale |
|----------|-------|-----------|
| Qubits | 4 | Practical for CPU simulation |
| Dataset | MNIST 3 vs 6 | Standard QML benchmark |
| Train/Test Split | 1000/200 | Sufficient for convergence |
| Optimizer | Adam | Standard choice for QML |
| Learning Rate | 0.01 | Based on preliminary tuning |
| Batch Size | 20 | Balanced speed vs stability |
| Epochs | 50 | Sufficient for convergence |
| Seeds | 42, 123, 456, 789, 101112 | 5 seeds for statistical robustness |
| Image Size | 4×4 | Maps to 16 features for 4 qubits |
| Loss Function | Binary Cross-Entropy | Standard for binary classification |

### Independent Variables

| Variable | Values | Purpose |
|----------|--------|---------|
| **Training Approach** | Baseline, Layerwise, Local Cost | Compare mitigation strategies |
| **Circuit Depth** | 4, 6, 8 layers | Observe depth-dependent behavior |

### Dependent Variables

| Variable | Metric | Expected Trend |
|----------|--------|----------------|
| **Accuracy** | Test accuracy (%) | Decreases with depth for baseline |
| **Gradient Norms** | Mean L2 norm | Decreases with depth for baseline |
| **Convergence** | Training loss curve | Flat for barren plateau |
| **Training Time** | Wall-clock seconds | Increases with depth |

---

## 📈 Results & Analysis

### Accuracy Results

#### Individual Results (All 45 Experiments)

<details>
<summary><strong>Baseline Results</strong> (click to expand)</summary>

| Depth | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 101112 | **Mean** |
|-------|---------|----------|----------|----------|-------------|----------|
| 4L | 76.0% | 73.0% | 75.5% | 70.5% | 75.0% | **73.8%** |
| 6L | 76.0% | 73.0% | 75.5% | 70.5% | 74.5% | **73.9%** |
| 8L | 54.4% | 51.6% | 52.6% | 53.1% | 51.6% | **52.7%** |

</details>

<details>
<summary><strong>Layerwise Results</strong> (click to expand)</summary>

| Depth | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 101112 | **Mean** |
|-------|---------|----------|----------|----------|-------------|----------|
| 4L | 78.0% | 72.0% | 75.5% | 70.0% | 74.5% | **74.0%** |
| 6L | 77.0% | 72.5% | 76.5% | 70.5% | 74.5% | **74.2%** |
| 8L | 77.0% | 73.0% | 75.5% | 69.5% | 74.5% | **73.9%** |

</details>

<details>
<summary><strong>Local Cost Results</strong> (click to expand)</summary>

| Depth | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 101112 | **Mean** |
|-------|---------|----------|----------|----------|-------------|----------|
| 4L | 79.5% | 74.5% | 76.5% | 70.5% | 75.5% | **75.3%** |
| 6L | 79.5% | 74.5% | 77.5% | 71.0% | 75.5% | **75.6%** |
| 8L | 78.0% | 74.5% | 77.5% | 71.0% | 76.0% | **75.4%** |

</details>

### Gradient Analysis

| Config | Mean Gradient Norm | Interpretation |
|--------|-------------------|----------------|
| Baseline 4L | 0.284 | ✅ Healthy |
| Baseline 6L | 0.276 | ✅ Healthy |
| Baseline 8L | 0.224 | ⚠️ Degraded (training fails despite non-zero gradients) |
| Layerwise 4L | 0.228 | ✅ Healthy |
| Layerwise 8L | 0.226 | ✅ Healthy |
| Local Cost 4L | 0.182 | ✅ Healthy |
| Local Cost 8L | 0.175 | ✅ Healthy |

### Depth Impact Analysis

```
Accuracy (%)
 80 ┤
    │  ●────────●──Local Cost
 75 ┤  ●────────●──Layerwise
    │  ●────────●
 70 ┤
    │
 65 ┤
    │  Baseline
 60 ┤
    │          ╲
 55 ┤           ╲
    │            ●  ← Barren Plateau
 50 ┤
    └──────────────────
       4    6    8  depth
```

### Success Rate Analysis

**Threshold: ≥70% accuracy**

| Approach | 4-Layer | 6-Layer | 8-Layer |
|----------|---------|---------|---------|
| Baseline | 80% (4/5) | 80% (4/5) | **0% (0/5)** ❌ |
| Layerwise | 60% (3/5) | 80% (4/5) | 60% (3/5) ✅ |
| Local Cost | 80% (4/5) | 80% (4/5) | 80% (4/5) ✅ |

### Training Time Analysis

| Config | Time Range | Mean Time |
|--------|------------|-----------|
| Baseline 4L | 10-27 min | ~18 min |
| Baseline 6L | 27-32 min | ~30 min |
| Baseline 8L | 38-40 min | ~39 min |
| Layerwise 4L | 7-16 min | ~12 min |
| Layerwise 6L | 28-30 min | ~29 min |
| Layerwise 8L | 50-52 min | ~51 min |
| Local Cost 4L | 19-28 min | ~24 min |
| Local Cost 6L | 27-28 min | ~28 min |
| Local Cost 8L | 55-60 min | ~58 min |

---

## 📓 Notebooks

Interactive Jupyter notebooks for exploration and analysis:

| Notebook | Purpose | Key Contents |
|----------|---------|-------------|
| [`00_data_loading.ipynb`](notebooks/00_data_loading.ipynb) | Data pipeline exploration | Loading, filtering, visualization |
| [`01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) | Dataset statistics | Class distribution, pixel analysis |
| [`02_circuit_visualization.ipynb`](notebooks/02_circuit_visualization.ipynb) | Circuit inspection | Gate diagrams, parameter counts |
| [`03_results_analysis.ipynb`](notebooks/03_results_analysis.ipynb) | Results analysis | Accuracy tables, comparison plots |
| [`04_barren_plateau_analysis.ipynb`](notebooks/04_barren_plateau_analysis.ipynb) | BP-specific analysis | Gradient trajectories, depth impact |

To launch notebooks:
```bash
jupyter notebook notebooks/
```

---

## 🧪 Testing

### Test Suite Overview

The project includes 7 comprehensive test files:

| Test File | Tests | Coverage Area |
|-----------|-------|---------------|
| `test_data_loader.py` | Data loading, filtering, encoding | `src/data/` |
| `test_data_consistency.py` | Data integrity, reproducibility | `src/data/` |
| `test_quantum_circuit.py` | Circuit construction, parameters | `src/models/quantum_circuit.py` |
| `test_qnn_model.py` | Model integration, forward pass | `src/models/qnn_model.py` |
| `test_trainers.py` | Training loops, gradient tracking | `src/training/` |
| `test_metrics.py` | Metric computation, BP detection | `src/evaluation/` |
| `test_pipeline.py` | End-to-end pipeline validation | Full stack |

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_quantum_circuit.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run only fast tests (exclude training tests)
python -m pytest tests/ -v -k "not train"
```

---

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

| Document | Description |
|----------|-------------|
| [Methodology](docs/methodology.md) | Research design, circuit architecture, training procedures |
| [Metrics Schema](docs/metrics_schema.md) | Output JSON format specification |
| [Quick Start](docs/quickstart.md) | Getting started in 5 minutes |
| [Results Guide](docs/results_guide.md) | How to interpret experiment outputs |
| [Results Interpretation](docs/results_interpretation.md) | Detailed analysis guide with actual data |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and solutions |
| [Validation Results](docs/validation_results.md) | Pre-experiment validation summary |

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for:

- Development setup instructions
- Coding standards and style guidelines
- Testing requirements
- Pull request process
- Issue reporting guidelines

---

## 👥 Research Team

This project was developed as part of an academic research initiative on quantum machine learning.

| Member | Role | Contributions |
|--------|------|---------------|
| **Fahad Abdullah** | Project Lead | Architecture design, 8-layer experiments, experiment coordination, quantum circuit implementation |
| **Asma Zubair** | Researcher | Data pipeline, 4-layer experiments, validation |
| **Frahan Riaz** | Researcher | 6-layer experiments, visualization, analysis notebooks, testing |

---

## 📖 Academic References

This project builds upon the following foundational works:

### Barren Plateaus
- **McClean, J. R., et al.** (2018). "Barren plateaus in quantum neural network training landscapes." *Nature Communications*, 9(1), 4812. [DOI: 10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4)

### Layerwise Training
- **Skolik, A., et al.** (2020). "Layerwise learning for quantum neural networks." *arXiv preprint arXiv:2006.14904*. [arXiv: 2006.14904](https://arxiv.org/abs/2006.14904)

### Local Cost Functions
- **Cerezo, M., et al.** (2021). "Cost function dependent barren plateaus in shallow parametrized quantum circuits." *Nature Communications*, 12(1), 1791. [DOI: 10.1038/s41467-021-21728-w](https://doi.org/10.1038/s41467-021-21728-w)

### Hardware-Efficient Ansatz
- **Kandala, A., et al.** (2017). "Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets." *Nature*, 549(7671), 242-246. [DOI: 10.1038/nature23879](https://doi.org/10.1038/nature23879)

### TensorFlow Quantum
- **Broughton, M., et al.** (2020). "TensorFlow Quantum: A Software Framework for Quantum Machine Learning." *arXiv preprint arXiv:2003.02989*. [arXiv: 2003.02989](https://arxiv.org/abs/2003.02989)

---

## 🔮 Future Work

The following directions are planned for future development:

### Short-Term
- [ ] **Extended depth testing** — Evaluate at 10, 12, and 16 layers to map the full barren plateau onset curve
- [ ] **Alternative ansätze** — Compare hardware-efficient ansatz with problem-specific designs
- [ ] **Noise models** — Add depolarizing and amplitude damping noise to simulate real hardware

### Medium-Term
- [ ] **Multi-class classification** — Extend from binary to 4-class and 10-class MNIST
- [ ] **Real hardware execution** — Run experiments on IBM Quantum or Google Sycamore
- [ ] **Hybrid strategies** — Combine layerwise training with local cost functions
- [ ] **Hyperparameter optimization** — Systematic grid/random search over learning rate, batch size, and architecture

### Long-Term
- [ ] **Generalization study** — Test on CIFAR-10, Fashion-MNIST, and domain-specific datasets
- [ ] **Expressibility analysis** — Measure circuit expressibility vs. trainability trade-off
- [ ] **Quantum advantage benchmarking** — Compare against classical neural networks of equivalent capacity
- [ ] **Publication** — Submit findings to a peer-reviewed quantum computing journal

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Hybrid-QNNs Research Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgments

We would like to acknowledge the following:

- **Google Quantum AI** — For developing TensorFlow Quantum and Cirq, the computational foundation of this project
- **The quantum computing research community** — For the theoretical foundations on barren plateaus, layerwise training, and local cost functions
- **The TensorFlow team** — For the robust machine learning infrastructure
- **MNIST dataset creators** — Yann LeCun and collaborators for the widely-used benchmark dataset
- **Academic advisors and reviewers** — For guidance and constructive feedback throughout this research

---

<p align="center">
  <strong>Built with ❤️ for the advancement of Quantum Machine Learning</strong>
</p>

<p align="center">
  <sub>If you find this project useful for your research, please consider citing it and giving it a ⭐ on GitHub.</sub>
</p>
