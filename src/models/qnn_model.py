"""Quantum Neural Network model implementation.

Provides TensorFlow Keras-compatible QNN models for binary classification
using parameterized quantum circuits (PQCs):

- ``QuantumNeuralNetwork``: standard end-to-end model (single PQC + fixed
  ``(x+1)/2`` output head; no trainable classical layers).
- ``LayerwiseQNN``: Skolik layerwise training with **true weight reuse**. At
  stage ``k`` the circuit bakes the already-trained layers ``0..k-1`` in
  as numeric rotation constants and exposes only layer ``k``'s ``2n`` symbolic
  parameters; trained values are persisted via ``store_current_params`` and
  the full fine-tune model is re-initialized exactly from the staged values.

References:
    - Layerwise training: Skolik et al., Quantum Machine Intelligence 3(5) (2021)
"""

import os
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')  # TFQ requires Keras 2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
from typing import Optional, List
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.quantum_circuit import QuantumCircuit, create_readout_operators

# Documented parameter-initialization distribution.
_PARAM_INIT_MIN = -0.05
_PARAM_INIT_MAX = 0.05


def _pqc_initializer(seed: Optional[int]):
    """Seeded PQC parameter initializer, or None for the TFQ default."""
    if seed is None:
        return None
    return tf.keras.initializers.RandomUniform(
        minval=_PARAM_INIT_MIN, maxval=_PARAM_INIT_MAX, seed=seed
    )


def _layer_symbols(layer: int, n_qubits: int) -> List[sympy.Symbol]:
    """Symbols for one ansatz layer in (qubit-major, ry then rz) order.

    Ordering matches ``QuantumCircuit._create_parameters`` so per-layer
    values align with ``QuantumCircuit.get_layer_parameters``.
    """
    symbols = []
    for q in range(n_qubits):
        symbols.append(sympy.Symbol(f'theta_{layer}_{q}_ry'))
        symbols.append(sympy.Symbol(f'theta_{layer}_{q}_rz'))
    return symbols


def _force_build(model: tf.keras.Model) -> None:
    """Force variable creation with a tiny dummy batch (empty circuits)."""
    if model.built:
        return
    dummy = [cirq.Circuit() for _ in range(2)]
    _ = model(tfq.convert_to_tensor(dummy), training=False)


class QuantumNeuralNetwork(tf.keras.Model):
    """Quantum Neural Network for binary classification.

    A single PQC over the cost readout ``PauliSum`` (global ``Z⊗…⊗Z`` or
    local ``(1/n)ΣᵢZᵢ``), terminated by the **fixed** output head
    ``Lambda((x + 1) / 2)``. There are **no trainable classical layers**: only
    the PQC parameters are trainable. Because the readout is a single
    ``PauliSum`` in both cost variants, the PQC output is ``(batch, 1)``
    identically.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 4,
        cost: str = 'global',
        init_seed: Optional[int] = None,
        name: str = "QNN"
    ):
        """
        Initialize Quantum Neural Network.

        Args:
            n_qubits: Number of qubits
            n_layers: Number of circuit layers
            cost: 'global' (product Z⊗…⊗Z) or 'local' ((1/n)ΣZᵢ)
            init_seed: Seed for the PQC parameter initializer. If None,
                       the TFQ default initializer is used.
            name: Model name
        """
        if cost not in ('global', 'local'):
            raise ValueError(f"cost must be 'global' or 'local', got {cost!r}")
        super().__init__(name=name)

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.cost = cost

        # Build quantum circuit
        self.qc = QuantumCircuit(n_qubits=n_qubits, n_layers=n_layers)
        self.circuit = self.qc.get_circuit()

        # Single cost readout operator
        self.readout_op = create_readout_operators(n_qubits, local=(cost == 'local'))

        self.quantum_layer = tfq.layers.PQC(
            self.circuit,
            self.readout_op,
            differentiator=tfq.differentiators.ParameterShift(),
            initializer=_pqc_initializer(init_seed),
        )

        # Fixed output head: no trainable classical layer.
        self.output_layer = tf.keras.layers.Lambda(
            lambda x: (x + 1) / 2,  # Map [-1, 1] to [0, 1]
            name='output'
        )

    def call(self, inputs, training=None):
        """
        Forward pass.

        Args:
            inputs: Input quantum circuits (batch of encoded data)

        Returns:
            Predictions in range [0, 1], shape (batch, 1)
        """
        expectations = self.quantum_layer(inputs)
        return self.output_layer(expectations)

    def _build_once(self) -> None:
        """Force variable creation with a tiny dummy batch (empty circuits)."""
        _force_build(self)

    def get_num_parameters(self) -> int:
        """
        Runtime-derived number of trainable parameters.

        Read from the instantiated model's ``trainable_variables`` (never
        from config or manual arithmetic).
        """
        self._build_once()
        return int(sum(tf.size(v) for v in self.trainable_variables))


class LayerwiseQNN:
    """Skolik layerwise QNN with true weight reuse.

    Incrementally builds the hardware-efficient ansatz one layer at a time.
    At stage ``k``:

    * layers ``0..k-1`` are **baked** into the circuit as numeric rotation
      gates using the values stored in ``param_values`` (constants, hence
      frozen by construction),
    * layer ``k`` is added as fresh symbolic parameters — the only trainable
      parameters at this stage (exactly ``2 * n_qubits``).

    ``store_current_params()`` persists the trained layer's values; only then
    may the next layer be added. ``build_finetune_model()`` re-instantiates
    the full ansatz with all layers symbolic, initialized exactly from the
    staged values, ready for fine-tuning.

    Invariants (guarded by ``tests/test_layerwise.py``):
      (i)   a stored ``param_values[k]`` never changes afterwards;
      (ii)  fine-tune initialization == the staged values;
      (iii) at stage ``k`` exactly ``2 * n_qubits`` parameters are trainable.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        target_layers: int = 4,
        cost: str = 'global',
        init_seed: Optional[int] = None
    ):
        """
        Initialize layerwise QNN.

        Args:
            n_qubits: Number of qubits
            target_layers: Target number of layers to build
            cost: 'global' (product Z⊗…⊗Z) or 'local' ((1/n)ΣZᵢ)
            init_seed: Base seed for per-layer parameter initialization; layer
                       ``k`` uses ``init_seed + k`` so stages are independent
                       draws while remaining reproducible.
        """
        if cost not in ('global', 'local'):
            raise ValueError(f"cost must be 'global' or 'local', got {cost!r}")

        self.n_qubits = n_qubits
        self.target_layers = target_layers
        self.cost = cost
        self.init_seed = init_seed
        self.qubits = cirq.GridQubit.rect(1, n_qubits)

        # Single cost readout operator
        self.readout_op = create_readout_operators(n_qubits, local=(cost == 'local'))

        # Per-layer trained values (length 2n each); None until trained.
        self.param_values: List[Optional[np.ndarray]] = [None] * target_layers
        self.current_layers = 0

        self.model: Optional[tf.keras.Model] = None
        self.circuit: Optional[cirq.Circuit] = None
        self.symbols: Optional[List[sympy.Symbol]] = None

    def add_layer(self) -> tf.keras.Model:
        """Build and return the stage-k model.

        Layers ``< k`` are baked in as numeric constants from ``param_values``;
        layer ``k`` is symbolic. Only ``2 * n_qubits`` parameters are trainable.
        """
        if self.current_layers >= self.target_layers:
            raise ValueError(f"Already at target number of layers ({self.target_layers})")
        k = self.current_layers

        circuit = cirq.Circuit()
        for layer in range(k):
            values = self.param_values[layer]
            if values is None:
                raise RuntimeError(
                    f"Cannot add layer {k}: layer {layer} has not been stored "
                    "(call store_current_params after training it)"
                )
            self._append_ansatz_layer(circuit, values=values)

        symbols = _layer_symbols(k, self.n_qubits)
        self._append_ansatz_layer(circuit, symbol=symbols)

        self.current_layers = k + 1
        self.circuit = circuit
        self.symbols = symbols
        self.model = self._create_model(
            circuit,
            name=f'QNN_L{self.current_layers}',
            init_seed=None if self.init_seed is None else self.init_seed + k,
        )
        return self.model

    def store_current_params(self) -> None:
        """Persist the current (newest) layer's trained values into ``param_values``."""
        if self.model is None:
            raise RuntimeError("No model to store; call add_layer first")
        k = self.current_layers - 1
        _force_build(self.model)
        values = self.model.trainable_variables[0].numpy()
        self.param_values[k] = np.array(values, dtype=np.float32, copy=True)

    def build_finetune_model(self) -> tf.keras.Model:
        """Build the full-ansatz fine-tune model initialized from staged values.

        All layers are symbolic; the parameter vector is set to
        ``concat(param_values)`` so fine-tuning starts exactly where staged
        training left off (invariant ii).
        """
        missing = [i for i, v in enumerate(self.param_values) if v is None]
        if missing:
            raise RuntimeError(
                f"All layers must be trained before fine-tuning; missing: {missing}"
            )

        circuit = cirq.Circuit()
        symbols: List[sympy.Symbol] = []
        for layer in range(self.target_layers):
            layer_syms = _layer_symbols(layer, self.n_qubits)
            symbols.extend(layer_syms)
            self._append_ansatz_layer(circuit, symbol=layer_syms)

        model = self._create_model(circuit, name='QNN_finetune', init_seed=None)
        self.model = model
        self.circuit = circuit
        self.symbols = symbols

        _force_build(model)
        flat = np.concatenate(self.param_values).astype(np.float32)
        model.set_weights([flat])
        return model

    def get_current_model(self) -> Optional[tf.keras.Model]:
        """Return the current model (last built), or None before any add_layer."""
        return self.model

    def _append_ansatz_layer(
        self,
        circuit: cirq.Circuit,
        symbol: Optional[List[sympy.Symbol]] = None,
        values: Optional[np.ndarray] = None,
    ) -> None:
        """Append one ansatz layer: RY/RZ per qubit + linear CNOT chain.

        Exactly one of ``symbol`` (symbolic rotations) or ``values`` (numeric
        baked rotations) must be provided. Mirrors ``QuantumCircuit``'s
        hardware-efficient ansatz structure.
        """
        if (symbol is None) == (values is None):
            raise ValueError("Exactly one of 'symbol' or 'values' must be set")
        for q in range(self.n_qubits):
            if values is not None:
                circuit.append(cirq.ry(values[2 * q])(self.qubits[q]))
                circuit.append(cirq.rz(values[2 * q + 1])(self.qubits[q]))
            else:
                circuit.append(cirq.ry(symbol[2 * q])(self.qubits[q]))
                circuit.append(cirq.rz(symbol[2 * q + 1])(self.qubits[q]))
        for i in range(self.n_qubits - 1):
            circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i + 1]))

    def _create_model(
        self,
        circuit: cirq.Circuit,
        name: str,
        init_seed: Optional[int],
    ) -> tf.keras.Model:
        """Build a Keras model: PQC (single PauliSum readout) + fixed head.

        The PQC layer resolves the circuit's symbols automatically and orders
        its kernel by sorted symbol name (layer-major, then qubit-major), which
        matches the ``param_values`` layout.
        """
        inputs = tf.keras.Input(shape=(), dtype=tf.string, name='circuits')
        quantum_layer = tfq.layers.PQC(
            circuit,
            self.readout_op,
            differentiator=tfq.differentiators.ParameterShift(),
            initializer=_pqc_initializer(init_seed),
        )
        expectations = quantum_layer(inputs)
        output = tf.keras.layers.Lambda(
            lambda x: (x + 1) / 2,  # Map [-1, 1] to [0, 1]
            name='output'
        )(expectations)
        return tf.keras.Model(inputs=inputs, outputs=output, name=name)


if __name__ == "__main__":
    # Test standard QNN
    print("Testing standard QNN...")
    qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=4, cost='global', init_seed=42)
    print(f"Number of parameters: {qnn.get_num_parameters()}")

    # Test layerwise QNN
    print("\nTesting layerwise QNN...")
    layerwise_qnn = LayerwiseQNN(n_qubits=4, target_layers=4, cost='global', init_seed=42)
    for i in range(4):
        model = layerwise_qnn.add_layer()
        layerwise_qnn.store_current_params()
        print(f"Layer {i + 1} added, model: {model.name}")
    finetune = layerwise_qnn.build_finetune_model()
    print(f"Finetune model: {finetune.name}, params={finetune.trainable_variables[0].numpy().shape}")