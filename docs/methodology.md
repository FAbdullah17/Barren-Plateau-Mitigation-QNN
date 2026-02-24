# Methodology Documentation

## Research Overview

This project implements and empirically compares three strategies for mitigating barren plateaus in quantum neural networks (QNNs):

1. **Baseline**: Standard end-to-end training
2. **Layerwise Training**: Incremental layer-by-layer optimization (Skolik et al., 2020)
3. **Local Cost Functions**: Per-qubit measurement operators (Cerezo et al., 2021)

---

## Barren Plateau Phenomenon

### Definition
A **barren plateau** occurs when gradients of a quantum circuit's cost function vanish exponentially with the number of qubits, making gradient-based optimization ineffective.

### Mathematical Formulation
For a parameterized quantum circuit $U(\theta)$ and cost function $C(\theta)$:

$$\mathbb{V}[\partial_{\theta_i} C] \propto \frac{1}{2^{n}}$$

where $n$ is the number of qubits.

### Causes
- **Circuit depth**: Deeper circuits experience more pronounced gradient vanishing
- **Global cost functions**: Measuring the entire quantum state
- **Random initialization**: Poor initialization can lead to flat loss landscapes

---

## Approach 1: Baseline Training

### Description
Standard end-to-end training of the full quantum circuit.

### Algorithm
1. Initialize all circuit parameters randomly
2. For each epoch:
   - Forward pass through complete circuit
   - Compute loss on global measurement
   - Backpropagate gradients
   - Update all parameters simultaneously

### Pseudocode
```python
model = QuantumNeuralNetwork(n_qubits, n_layers)
optimizer = Adam(learning_rate)

for epoch in range(epochs):
    for batch in data_loader:
        predictions = model(batch.X)
        loss = binary_crossentropy(predictions, batch.y)
        gradients = compute_gradients(loss, model.parameters)
        optimizer.apply_gradients(gradients)
```

### Advantages
- Simple implementation
- Standard ML workflow
- No architectural constraints

### Disadvantages
- Susceptible to barren plateaus at depth
- Gradients vanish exponentially
- Poor convergence for deep circuits

---

## Approach 2: Layerwise Training

### Description
Based on Skolik et al. (2020), train circuits layer-by-layer, adding and optimizing one layer at a time.

### Algorithm
1. Start with 1-layer circuit
2. Train layer 1 for `epochs_per_layer` epochs
3. **Freeze** layer 1 parameters
4. Add layer 2, train for `epochs_per_layer` epochs
5. Repeat until all layers added
6. **Fine-tune** all layers together for `finetune_epochs` epochs

### Pseudocode
```python
qnn = LayerwiseQNN(n_qubits, total_layers)

for layer_idx in range(1, total_layers + 1):
    qnn.add_layer()
    
    # Train only new layer (previous frozen)
    for epoch in range(epochs_per_layer):
        train_step(qnn, data, optimizer)
    
    qnn.freeze_previous_layers()

# Fine-tuning phase
qnn.unfreeze_all_layers()
for epoch in range(finetune_epochs):
    train_step(qnn, data, optimizer)
```

### Key Implementation Details
- **Freezing**: Set `requires_grad=False` for previous layer parameters
- **Gradual depth increase**: Avoids deep circuit training initially
- **Fine-tuning**: Allows cross-layer optimization after layerwise training

### Theoretical Justification
- Shallower circuits during initial training have larger gradients
- Each layer learns on top of features from previous layers
- Reduces effective depth during critical training phases

### Advantages
- Mitigates gradient vanishing
- Better gradient flow in early training
- More stable optimization

### Disadvantages
- Longer total training time
- More hyperparameters (epochs_per_layer, finetune_epochs)
- Layer order dependency

---

## Approach 3: Local Cost Functions

### Description
Based on Cerezo et al. (2021), use per-qubit measurement operators instead of global state measurement.

### Mathematical Formulation

**Global Cost:**
$$C_{\text{global}}(\theta) = \langle \psi(\theta) | \hat{O}_{\text{global}} | \psi(\theta) \rangle$$

where $\hat{O}_{\text{global}}$ measures the entire quantum state.

**Local Cost:**
$$C_{\text{local}}(\theta) = \sum_{i=1}^{n} \langle \psi(\theta) | \hat{O}_i | \psi(\theta) \rangle$$

where $\hat{O}_i = Z_i$ measures qubit $i$ individually.

### Implementation
```python
# Global cost (default) — single Pauli-Z on first qubit
model = QuantumNeuralNetwork(n_qubits=4, local_cost=False)
readout_ops = [cirq.Z(q0)]

# Local cost — independent Pauli-Z on each qubit
model = QuantumNeuralNetwork(n_qubits=4, local_cost=True)
readout_ops = [cirq.Z(q0), cirq.Z(q1), cirq.Z(q2), cirq.Z(q3)]
```

### Theoretical Justification
- Local operators reduce correlation length
- Gradient variance scales as $1/\text{poly}(n)$ instead of $1/2^n$
- Preserves trainability at depth

### Advantages
- Maintains gradient scale at depth
- Simple modification to baseline
- No architectural changes needed

### Disadvantages
- May lose some expressivity
- Not suitable for all tasks
- Requires task-compatible local measurements

---

## Hardware-Efficient Ansatz

### Circuit Structure
Each layer consists of:

1. **Single-qubit rotations**: RY and RZ gates on each qubit
2. **Entangling gates**: CNOT gates in linear topology

### Mathematical Representation
Layer $\ell$:

$$U_\ell(\theta) = \text{CNOT}_{\text{linear}} \cdot \prod_{i=1}^{n} RZ(\theta_{\ell,i}^z) RY(\theta_{\ell,i}^y)$$

### Parameter Count
For $n$ qubits and $L$ layers:
$$\text{Parameters} = 2 \times n \times L$$

### Entanglement Topology
Linear nearest-neighbor:
```
q0 --●--    --●--
     |        |
q1 --⊕--●--  |
        |    |
q2 -----⊕--● |
           | |
q3 --------⊕-●
```

---

## Data Preprocessing

### MNIST Binary Classification
- **Task**: Classify digits 3 vs 6
- **Original size**: 28×28 = 784 pixels
- **Downsampling**: 4×4 = 16 pixels (via bilinear interpolation)
- **Normalization**: [0, 1] range (min-max normalization)
- **Encoding**: Angle encoding via RY rotations: `RY(x_i × π)`

### Quantum Encoding
Classical data $x \in \mathbb{R}^{16}$ encoded as:

$$|\psi(x)\rangle = U_{\text{enc}}(x)|0\rangle^{\otimes 4}$$

where $U_{\text{enc}}(x) = \prod_{i=1}^{4} RY(\arcsin(x_i))$

---

## Training Procedure

### Loss Function
Binary Cross-Entropy (BCE):

$$L(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

### Optimizer
Adam optimizer with:
- Learning rate: 0.01
- Beta1: 0.9
- Beta2: 0.999
- Epsilon: 1e-7

### Batching
- Batch size: 20 samples
- Shuffle: True
- Drop last: False

### Gradient Computation
TensorFlow Quantum's parameter-shift rule for gradient calculation.

---

## Evaluation Metrics

### Primary Metrics
1. **Test Accuracy**: Classification accuracy on held-out test set
2. **Training Time**: Wall-clock time for training
3. **Gradient Norms**: $\|\nabla_\theta L\|_2$ at each epoch

### Gradient Statistics
- **Mean gradient norm**: $\mu_g = \mathbb{E}[\|\nabla_\theta L\|_2]$
- **Gradient variance**: $\sigma_g^2 = \text{Var}[\|\nabla_\theta L\|_2]$

### Barren Plateau Detection
Flag barren plateau if:

$$\mu_g < \tau \quad \text{for} \quad \tau = 10^{-6}$$

### Success Rate
Percentage of runs achieving ≥90% test accuracy.

---

## Experimental Design

### Multi-Depth Comparison
- **Circuit depths**: 4, 6, 8 layers
- **Random seeds**: 42, 123, 456, 789, 101112
- **Total runs**: 3 approaches × 3 depths × 5 seeds = **45 experiments**

### Statistical Analysis
- **Mean ± Std**: Average accuracy across seeds
- **Success rate**: Percentage reaching threshold
- **t-tests**: Pairwise comparison between approaches
- **Effect size**: Cohen's d for practical significance

---

## Hyperparameters

### Fixed Across All Experiments
| Parameter | Value |
|-----------|-------|
| n_qubits | 4 |
| learning_rate | 0.01 |
| batch_size | 20 |
| digit1 | 3 |
| digit2 | 6 |
| train_size | 1000 |
| test_size | 200 |

### Approach-Specific

**Baseline & Local Cost:**
- epochs: 50

**Layerwise:**
- epochs_per_layer: 10
- finetune_epochs: 10
- Total: 10 × layers + 10 fine-tune

---

## Reproducibility

### Random Seed Management
- Data loading: seed controls train/test split
- Model initialization: seed sets initial parameters
- Training: seed determines batch ordering

### Version Control
- Python: 3.10
- TensorFlow: 2.15.0
- TensorFlow Quantum: 0.7.2
- Cirq: 1.3.0

### Hardware Requirements
- CPU-only execution (no GPU required for 4-qubit circuits)
- Memory: ~4 GB RAM
- Storage: ~500 MB for results

---

## References

1. **Skolik, A., McClean, J. R., Mohseni, M., van der Smagt, P., & Leib, M.** (2020). Layerwise learning for quantum neural networks. *Quantum Machine Intelligence*, 3(1), 1-11.

2. **Cerezo, M., Sone, A., Volkoff, T., Cincio, L., & Coles, P. J.** (2021). Cost function dependent barren plateaus in shallow parametrized quantum circuits. *Nature Communications*, 12(1), 1791.

3. **McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., & Neven, H.** (2018). Barren plateaus in quantum neural network training landscapes. *Nature Communications*, 9(1), 4812.
