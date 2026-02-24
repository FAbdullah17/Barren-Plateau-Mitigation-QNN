"""Quantum circuit construction for hardware-efficient ansatz.

Provides parameterized quantum circuit building using RY/RZ rotations
and CNOT entangling gates in a linear topology. Supports variable qubit
counts and circuit depths with per-layer parameter access for layerwise
training strategies.

References:
    - Hardware-efficient ansatz: Kandala et al., Nature 549, 242-246 (2017)
"""

import cirq
import sympy
from typing import List, Tuple
import numpy as np


class QuantumCircuit:
    """Hardware-efficient ansatz for quantum neural networks."""
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 4):
        """
        Initialize quantum circuit.
        
        Args:
            n_qubits: Number of qubits (default: 4)
            n_layers: Number of parameterized layers (default: 4)
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        self.params = self._create_parameters()
        self.circuit = self._build_circuit()
        
    def _create_parameters(self) -> List[List[sympy.Symbol]]:
        """Create symbolic parameters for each layer and qubit."""
        params = []
        for layer in range(self.n_layers):
            layer_params = []
            for qubit in range(self.n_qubits):
                # Two rotation parameters per qubit per layer (RY, RZ)
                ry_param = sympy.Symbol(f'theta_{layer}_{qubit}_ry')
                rz_param = sympy.Symbol(f'theta_{layer}_{qubit}_rz')
                layer_params.extend([ry_param, rz_param])
            params.append(layer_params)
        return params
    
    def _build_circuit(self) -> cirq.Circuit:
        """Build the full hardware-efficient ansatz circuit."""
        circuit = cirq.Circuit()
        
        for layer in range(self.n_layers):
            # Parameterized single-qubit rotation gates
            for i, qubit in enumerate(self.qubits):
                ry_param = self.params[layer][2*i]
                rz_param = self.params[layer][2*i + 1]
                circuit.append([
                    cirq.ry(ry_param)(qubit),
                    cirq.rz(rz_param)(qubit)
                ])
            
            # Entangling layer (CNOT gates in linear topology)
            for i in range(self.n_qubits - 1):
                circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i+1]))
        
        return circuit
    
    def get_circuit(self) -> cirq.Circuit:
        """Return the full quantum circuit."""
        return self.circuit
    
    def get_parameters(self) -> List[sympy.Symbol]:
        """Get all circuit parameters as a flat list."""
        flat_params = []
        for layer_params in self.params:
            flat_params.extend(layer_params)
        return flat_params
    
    def get_layer_parameters(self, layer_idx: int) -> List[sympy.Symbol]:
        """Get parameters for a specific layer.
        
        Args:
            layer_idx: Index of the target layer (0-based).
            
        Returns:
            List of sympy.Symbol parameters for the specified layer.
            
        Raises:
            ValueError: If layer_idx is out of range.
        """
        if layer_idx >= self.n_layers:
            raise ValueError(f"Layer index {layer_idx} out of range (max: {self.n_layers-1})")
        return self.params[layer_idx]
    
    def get_circuit_up_to_layer(self, layer_idx: int) -> cirq.Circuit:
        """
        Get circuit containing only layers from 0 to layer_idx (inclusive).
        
        Used in layerwise training to incrementally build circuit depth.
        
        Args:
            layer_idx: Last layer to include (0-based, inclusive).
            
        Returns:
            Circuit with the specified number of layers.
        """
        circuit = cirq.Circuit()
        
        for layer in range(layer_idx + 1):
            # Parameterized single-qubit rotation gates
            for i, qubit in enumerate(self.qubits):
                ry_param = self.params[layer][2*i]
                rz_param = self.params[layer][2*i + 1]
                circuit.append([
                    cirq.ry(ry_param)(qubit),
                    cirq.rz(rz_param)(qubit)
                ])
            
            # Entangling layer
            for i in range(self.n_qubits - 1):
                circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i+1]))
        
        return circuit
    
    def visualize(self) -> str:
        """Return text representation of the circuit."""
        return str(self.circuit)


def create_readout_operators(n_qubits: int, local: bool = False) -> List[cirq.PauliSum]:
    """
    Create measurement operators for the QNN readout layer.
    
    Args:
        n_qubits: Number of qubits in the circuit.
        local: If True, create per-qubit Pauli-Z operators for local
               cost function training (Cerezo et al., 2021).
               If False, create a single global Pauli-Z operator on
               the first qubit.
               
    Returns:
        List of Pauli-Z measurement operators.
    """
    qubits = cirq.GridQubit.rect(1, n_qubits)
    
    if local:
        # Local cost: independent measurement on each qubit
        operators = [cirq.Z(qubit) for qubit in qubits]
    else:
        # Global cost: measure first qubit only
        operators = [cirq.Z(qubits[0])]
    
    return operators


if __name__ == "__main__":
    # Verify circuit construction
    qc = QuantumCircuit(n_qubits=4, n_layers=4)
    print("Circuit parameters:", len(qc.get_parameters()))
    print("\nCircuit visualization:")
    print(qc.visualize())
    
    # Verify layer extraction for layerwise training
    layer_2_circuit = qc.get_circuit_up_to_layer(1)
    print(f"\nCircuit up to layer 1 has {len(layer_2_circuit)} moments")
