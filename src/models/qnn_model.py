"""Quantum Neural Network model implementation.

Provides TensorFlow Keras-compatible QNN models for binary classification
using parameterized quantum circuits (PQCs). Includes both standard
end-to-end training (QuantumNeuralNetwork) and incremental layerwise
training (LayerwiseQNN) architectures.

References:
    - Layerwise training: Skolik et al., Quantum Machine Intelligence 3(5) (2021)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import numpy as np
from typing import Optional, List
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.quantum_circuit import QuantumCircuit, create_readout_operators


class QuantumNeuralNetwork(tf.keras.Model):
    """Quantum Neural Network for binary classification."""
    
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 4,
        local_cost: bool = False,
        name: str = "QNN"
    ):
        """
        Initialize Quantum Neural Network.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of circuit layers
            local_cost: Use local cost functions (per-qubit measurements)
            name: Model name
        """
        super().__init__(name=name)
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.local_cost = local_cost
        
        # Build quantum circuit
        self.qc = QuantumCircuit(n_qubits=n_qubits, n_layers=n_layers)
        self.circuit = self.qc.get_circuit()
        
        # Create readout operators
        self.readout_ops = create_readout_operators(n_qubits, local=local_cost)
        
        # Create TFQ layer
        self.quantum_layer = tfq.layers.PQC(
            self.circuit,
            self.readout_ops,
            differentiator=tfq.differentiators.ParameterShift()
        )
        
        # Output processing
        if local_cost:
            # Average local measurements
            self.output_layer = tf.keras.layers.Dense(
                1, activation='sigmoid', name='output'
            )
        else:
            # Single global measurement
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
            Predictions in range [0, 1]
        """
        # Get quantum expectations
        expectations = self.quantum_layer(inputs)
        
        if self.local_cost:
            # Average local expectations and pass through dense layer
            output = self.output_layer(expectations)
        else:
            # Process global expectation
            output = self.output_layer(expectations)
        
        return output
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return len(self.qc.get_parameters())


class LayerwiseQNN:
    """
    Quantum Neural Network with layerwise training capability.
    
    This class manages incremental layer-by-layer training as described
    in Skolik et al. (2020).
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        target_layers: int = 4,
        local_cost: bool = False
    ):
        """
        Initialize layerwise QNN.
        
        Args:
            n_qubits: Number of qubits
            target_layers: Target number of layers to build
            local_cost: Use local cost functions
        """
        self.n_qubits = n_qubits
        self.target_layers = target_layers
        self.local_cost = local_cost
        self.current_layers = 0
        
        # Build full circuit for reference
        self.full_qc = QuantumCircuit(n_qubits=n_qubits, n_layers=target_layers)
        self.readout_ops = create_readout_operators(n_qubits, local=local_cost)
        
        # Current model
        self.model = None
        self.trained_params = []
    
    def add_layer(self) -> tf.keras.Model:
        """
        Add and return a new layer to the model.
        
        Returns:
            Model with one additional layer
        """
        if self.current_layers >= self.target_layers:
            raise ValueError("Already at target number of layers")
        
        self.current_layers += 1
        
        # Get circuit up to current layer
        circuit = self.full_qc.get_circuit_up_to_layer(self.current_layers - 1)
        
        # Create new model
        self.model = self._create_model_for_circuit(circuit)
        
        return self.model
    
    def _create_model_for_circuit(self, circuit: cirq.Circuit) -> tf.keras.Model:
        """Create TensorFlow model for given circuit."""
        inputs = tf.keras.Input(shape=(), dtype=tf.string, name='circuits')
        
        # Quantum layer
        quantum_layer = tfq.layers.PQC(
            circuit,
            self.readout_ops,
            differentiator=tfq.differentiators.ParameterShift()
        )
        expectations = quantum_layer(inputs)
        
        # Output processing
        if self.local_cost:
            output = tf.keras.layers.Dense(1, activation='sigmoid')(expectations)
        else:
            output = tf.keras.layers.Lambda(lambda x: (x + 1) / 2)(expectations)
        
        model = tf.keras.Model(inputs=inputs, outputs=output, name=f'QNN_L{self.current_layers}')
        return model
    
    def get_current_model(self) -> Optional[tf.keras.Model]:
        """Get current model."""
        return self.model
    
    def freeze_trained_layers(self):
        """Freeze parameters of already trained layers."""
        # This is handled in the training loop by selective parameter updates
        pass


if __name__ == "__main__":
    # Test standard QNN
    print("Testing standard QNN...")
    qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=4, local_cost=False)
    print(f"Number of parameters: {qnn.get_num_parameters()}")
    
    # Test layerwise QNN
    print("\nTesting layerwise QNN...")
    layerwise_qnn = LayerwiseQNN(n_qubits=4, target_layers=4)
    for i in range(4):
        model = layerwise_qnn.add_layer()
        print(f"Layer {i+1} added, model: {model.name}")
