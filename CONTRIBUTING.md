# Contributing to Barren-Plateau-Mitigation-QNN

Thank you for considering contributing to the **Barren Plateau Mitigation in Quantum Neural Networks** project! This document provides comprehensive guidelines and best practices for contributing to this research codebase.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Contribution Workflow](#contribution-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Standards](#documentation-standards)
- [Configuration Files](#configuration-files)
- [Experiment Contributions](#experiment-contributions)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Review Process](#review-process)

---

## Code of Conduct

This project follows a professional academic research environment. All contributors are expected to:

- Be respectful and constructive in all interactions
- Provide evidence-based technical feedback
- Acknowledge the work of others appropriately
- Maintain scientific integrity and reproducibility
- Follow the established coding and documentation standards

---

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Python 3.10** (required for TensorFlow Quantum compatibility)
- **Git** for version control
- **Virtual environment** support (venv or conda)
- Basic understanding of quantum computing and machine learning concepts

### First-Time Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/Barren-Plateau-Mitigation-QNN.git
   cd Barren-Plateau-Mitigation-QNN
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/FAbdullah17/Barren-Plateau-Mitigation-QNN.git
   ```
4. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate       # Linux/Mac
   .\venv\Scripts\activate        # Windows
   ```
5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Development Setup

### Environment Verification

After installation, verify your environment:

```bash
python -c "import tensorflow as tf; print(f'TF: {tf.__version__}')"
python -c "import tensorflow_quantum as tfq; print(f'TFQ: {tfq.__version__}')"
python -c "import cirq; print(f'Cirq: {cirq.__version__}')"
```

Expected output:
```
TF: 2.15.0
TFQ: 0.7.2
Cirq: 1.3.0
```

### Running Tests

Before making changes, ensure all tests pass:

```bash
python -m pytest tests/ -v
```

---

## Project Structure

Understanding the project layout is essential before contributing:

```
Barren-Plateau-Mitigation-QNN/
├── src/                        # Core source code
│   ├── data/                   # Data loading and preprocessing
│   │   └── mnist_loader.py     # MNIST binary classification loader
│   ├── models/                 # Quantum circuit and model definitions
│   │   ├── quantum_circuit.py  # Hardware-efficient ansatz
│   │   └── qnn_model.py       # TFQ-integrated QNN models
│   ├── training/               # Training strategies
│   │   ├── baseline_trainer.py # Standard end-to-end training
│   │   └── layerwise_trainer.py# Layerwise incremental training
│   ├── evaluation/             # Metrics and visualization
│   │   ├── metrics.py          # Gradient tracking, BP detection
│   │   └── visualization.py   # Plotting utilities
│   └── utils/                  # Shared utilities
│       └── logging_config.py   # Logging infrastructure
├── experiments/                # Experiment runners
│   ├── run_baseline.py         # Baseline experiment entry point
│   ├── run_layerwise.py        # Layerwise experiment entry point
│   ├── run_local_cost.py       # Local cost experiment entry point
│   └── run_comparison.py       # Cross-approach comparison
├── configs/                    # YAML experiment configurations
├── scripts/                    # Batch automation and analysis
├── tests/                      # Unit and integration tests
├── notebooks/                  # Jupyter analysis notebooks
├── docs/                       # Documentation
└── results/                    # Experiment outputs (git-ignored)
```

### Key Modules

| Module | File | Purpose |
|--------|------|---------|
| Data Loader | `src/data/mnist_loader.py` | MNIST download, downsampling, encoding |
| Quantum Circuit | `src/models/quantum_circuit.py` | Hardware-efficient ansatz construction |
| QNN Model | `src/models/qnn_model.py` | TFQ layer integration, readout operators |
| Baseline Trainer | `src/training/baseline_trainer.py` | Standard training with gradient tracking |
| Layerwise Trainer | `src/training/layerwise_trainer.py` | Incremental layer-by-layer training |
| Metrics | `src/evaluation/metrics.py` | Accuracy, gradient stats, BP detection |
| Visualization | `src/evaluation/visualization.py` | Training curves, comparison plots |

---

## Contribution Workflow

### Branch Naming Convention

Use descriptive branch names:

```
feature/<description>     # New functionality
fix/<description>         # Bug fixes
docs/<description>        # Documentation changes
experiment/<description>  # New experiment configurations
refactor/<description>    # Code restructuring
```

**Examples:**
```bash
git checkout -b feature/noise-model-support
git checkout -b fix/gradient-tracking-memory
git checkout -b docs/update-methodology
git checkout -b experiment/12-layer-depth-test
```

### Making Changes

1. **Create a feature branch** from `main`:
   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards below

3. **Run tests** to ensure nothing is broken:
   ```bash
   python -m pytest tests/ -v
   ```

4. **Commit with clear messages**:
   ```bash
   git add -A
   git commit -m "feat: add noise model support to quantum circuit"
   ```

5. **Push and create a Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short description>

<optional body>
```

**Types:**
| Type | Description |
|------|-------------|
| `feat` | New feature or functionality |
| `fix` | Bug fix |
| `docs` | Documentation only changes |
| `test` | Adding or modifying tests |
| `refactor` | Code change that neither adds feature nor fixes bug |
| `perf` | Performance improvement |
| `chore` | Maintenance tasks |

---

## Coding Standards

### Python Style

- Follow **PEP 8** conventions
- Maximum line length: **100 characters**
- Use **type hints** for function signatures
- Use **docstrings** for all public functions and classes

### Docstring Format

Use Google-style docstrings:

```python
def compute_gradient_stats(gradients: np.ndarray, threshold: float = 1e-6) -> dict:
    """Compute statistical summary of gradient norms.

    Analyzes gradient magnitudes to detect potential barren plateau
    conditions based on the provided threshold.

    Args:
        gradients: Array of gradient values with shape (n_epochs, n_params).
        threshold: Minimum gradient norm for barren plateau detection.
            Defaults to 1e-6.

    Returns:
        Dictionary containing gradient statistics:
            - mean_norm: Average gradient magnitude
            - std_norm: Standard deviation of gradient norms
            - variance: Variance of gradient norms
            - barren_plateau_detected: Whether gradients fell below threshold

    Raises:
        ValueError: If gradients array is empty.
    """
```

### Import Order

Follow this import ordering:

```python
# 1. Standard library
import os
import json
from pathlib import Path

# 2. Third-party libraries
import numpy as np
import tensorflow as tf
import cirq

# 3. Local imports
from src.data import load_mnist_binary
from src.models import QuantumCircuit
```

### Code Organization

- Keep functions focused and under 50 lines where possible
- Use meaningful variable names (no single letters except loop counters)
- Add inline comments for non-obvious logic
- Group related functionality into classes

---

## Testing Guidelines

### Test Structure

Tests are organized by module:

```
tests/
├── test_data_loader.py         # Data loading tests
├── test_data_consistency.py    # Data integrity tests
├── test_quantum_circuit.py     # Circuit construction tests
├── test_qnn_model.py          # Model integration tests
├── test_trainers.py           # Training strategy tests
├── test_metrics.py            # Evaluation metrics tests
└── test_pipeline.py           # End-to-end pipeline tests
```

### Writing Tests

- Use **pytest** as the testing framework
- Test file names must start with `test_`
- Test function names must start with `test_`
- Each test should verify one specific behavior
- Use descriptive test names that explain what is being tested

**Example:**

```python
def test_quantum_circuit_creates_correct_number_of_parameters():
    """Verify circuit parameter count matches n_qubits * n_layers * 3."""
    circuit = QuantumCircuit(n_qubits=4, n_layers=4)
    expected_params = 4 * 4 * 3  # qubits * layers * gates_per_qubit
    assert len(circuit.get_parameters()) == expected_params

def test_barren_plateau_detected_when_gradients_vanish():
    """Verify BP detection triggers when gradient norms fall below threshold."""
    tiny_gradients = np.random.uniform(0, 1e-8, size=(50, 48))
    stats = compute_gradient_stats(tiny_gradients, threshold=1e-6)
    assert stats['barren_plateau_detected'] is True
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_quantum_circuit.py -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=html
```

### Test Coverage Requirements

- All new code should include corresponding tests
- Aim for >80% test coverage on new modules
- Critical paths (training, gradient computation) require >90% coverage

---

## Documentation Standards

### Code Documentation

Every module must have:

1. **Module-level docstring** — Purpose, key classes/functions, and references
2. **Class docstrings** — Purpose, attributes, and usage
3. **Function docstrings** — Purpose, args, returns, and exceptions

### Markdown Documentation

When updating or creating documentation in `docs/`:

- Use clear section headings with `##` and `###`
- Include code examples where appropriate
- Cross-reference other documentation files
- Keep accuracy tables updated with actual experimental data
- Never include fabricated or placeholder statistics

### Academic References

When implementing algorithms from papers, always cite the source:

```python
"""
Layerwise training strategy for quantum neural networks.

Implements the incremental training approach proposed by Skolik et al.,
where layers are added and trained sequentially to mitigate barren plateaus.

References:
    Skolik, A., et al. (2020). "Layerwise learning for quantum neural networks."
    arXiv:2006.14904
"""
```

---

## Configuration Files

### YAML Config Structure

All experiment configurations follow a standardized schema:

```yaml
experiment:
  name: "<experiment_identifier>"
  approach: "<baseline|layerwise|local_cost>"
  description: "<brief description>"

model:
  n_qubits: 4
  n_layers: <4|6|8>
  circuit_type: "hardware_efficient"

training:
  optimizer: "adam"
  learning_rate: 0.01
  batch_size: 20
  epochs: 50
  local_cost: <true|false>

data:
  dataset: "mnist"
  digit1: 3
  digit2: 6
  train_size: 1000
  test_size: 200
  image_size: [4, 4]
```

### Adding New Configurations

1. Create the YAML file in `configs/`
2. Follow the naming convention: `<approach>_<depth>layer.yaml`
3. Include all required fields from the schema above
4. Add descriptive comments at the top of the file

---

## Experiment Contributions

### Adding a New Training Approach

1. Create the trainer in `src/training/new_trainer.py`
2. Follow the interface established by `BaselineTrainer`
3. Create the experiment runner in `experiments/run_new_approach.py`
4. Add configuration files in `configs/`
5. Write comprehensive tests in `tests/test_new_trainer.py`
6. Update documentation in `docs/methodology.md`

### Adding a New Analysis

1. Create analysis scripts in `scripts/`
2. Create visualization notebooks in `notebooks/`
3. Document the analysis methodology
4. Include example outputs

---

## Pull Request Process

### Before Submitting

- [ ] All tests pass (`python -m pytest tests/ -v`)
- [ ] Code follows PEP 8 style guidelines
- [ ] Docstrings added/updated for all new code
- [ ] Documentation updated if behavior changes
- [ ] No hardcoded paths or credentials
- [ ] Commit messages follow conventional format

### PR Description Template

```markdown
## Summary
Brief description of changes.

## Changes Made
- List of specific changes

## Testing
- How the changes were tested
- Test commands run

## Related Issues
- Closes #<issue_number>
```

### Review Criteria

Pull requests are reviewed for:

1. **Correctness** — Does the code work as intended?
2. **Testing** — Are changes adequately tested?
3. **Style** — Does the code follow project conventions?
4. **Documentation** — Are changes documented?
5. **Performance** — Are there any performance concerns?
6. **Reproducibility** — Can results be reproduced deterministically?

---

## Issue Reporting

### Bug Reports

When reporting bugs, include:

1. **Environment** — OS, Python version, TF version
2. **Steps to reproduce** — Exact commands to trigger the bug
3. **Expected behavior** — What should happen
4. **Actual behavior** — What actually happens
5. **Error traceback** — Full error output
6. **Configuration** — YAML config file used

### Feature Requests

When requesting features, describe:

1. **Use case** — Why is this needed?
2. **Proposed solution** — How should it work?
3. **Alternatives considered** — What other approaches were evaluated?
4. **Impact** — How does this affect existing functionality?

---

## Review Process

1. **Automated checks** — Tests must pass
2. **Peer review** — At least one team member reviews the code
3. **Documentation review** — Documentation accuracy is verified
4. **Merge** — Approved PRs are merged to `main`

---

## Questions?

If you have questions about contributing, please:

1. Check the existing [documentation](docs/)
2. Search for related [issues](https://github.com/FAbdullah17/Barren-Plateau-Mitigation-QNN/issues)
3. Open a new issue with the `question` label

Thank you for contributing to advancing quantum machine learning research! 🚀
