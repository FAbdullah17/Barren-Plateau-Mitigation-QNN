"""Unit tests for quantum circuit construction."""

import pytest
import cirq
import sympy
import numpy as np
from src.models.quantum_circuit import QuantumCircuit


class TestQuantumCircuit:
    """Test quantum circuit builder."""
    
    def test_initialization(self):
        """Test circuit initialization."""
        qc = QuantumCircuit(n_qubits=4, n_layers=3)
        
        assert qc.n_qubits == 4
        assert qc.n_layers == 3
        assert len(qc.qubits) == 4
        assert all(isinstance(q, cirq.GridQubit) for q in qc.qubits)
    
    def test_create_data_encoding_circuit(self):
        """Test data encoding circuit creation."""
        qc = QuantumCircuit(n_qubits=4, n_layers=3)
        encoding_circuit = qc.create_data_encoding_circuit()
        
        # Should have 4 parameters (one per qubit)
        symbols = list(encoding_circuit.all_operations())
        assert len(symbols) > 0
        
        # Check it's a valid Cirq circuit
        assert isinstance(encoding_circuit, cirq.Circuit)
    
    def test_create_variational_circuit(self):
        """Test variational circuit creation."""
        qc = QuantumCircuit(n_qubits=4, n_layers=3)
        var_circuit = qc.create_variational_circuit()
        
        # Check it's a valid circuit
        assert isinstance(var_circuit, cirq.Circuit)
        
        # Should have operations (gates)
        ops = list(var_circuit.all_operations())
        assert len(ops) > 0
        
        # Verify it uses the correct qubits
        used_qubits = set()
        for op in ops:
            used_qubits.update(op.qubits)
        assert len(used_qubits) <= 4
    
    def test_get_circuit(self):
        """Test full circuit retrieval."""
        qc = QuantumCircuit(n_qubits=4, n_layers=3)
        full_circuit = qc.get_circuit()
        
        assert isinstance(full_circuit, cirq.Circuit)
        assert len(list(full_circuit.all_operations())) > 0
    
    def test_get_circuit_up_to_layer(self):
        """Test layerwise circuit construction."""
        qc = QuantumCircuit(n_qubits=4, n_layers=4)
        
        # Get circuits up to different layers
        circuit_1 = qc.get_circuit_up_to_layer(1)
        circuit_2 = qc.get_circuit_up_to_layer(2)
        circuit_4 = qc.get_circuit_up_to_layer(4)
        
        # Check they're all valid
        assert isinstance(circuit_1, cirq.Circuit)
        assert isinstance(circuit_2, cirq.Circuit)
        assert isinstance(circuit_4, cirq.Circuit)
        
        # Layer 2 should have more operations than layer 1
        ops_1 = len(list(circuit_1.all_operations()))
        ops_2 = len(list(circuit_2.all_operations()))
        assert ops_2 >= ops_1
    
    def test_get_circuit_up_to_layer_invalid(self):
        """Test error handling for invalid layer numbers."""
        qc = QuantumCircuit(n_qubits=4, n_layers=3)
        
        with pytest.raises((ValueError, AssertionError)):
            qc.get_circuit_up_to_layer(0)  # Layer must be >= 1
        
        with pytest.raises((ValueError, AssertionError)):
            qc.get_circuit_up_to_layer(5)  # Exceeds n_layers
    
    def test_create_readout_operators_global(self):
        """Test global readout operator creation."""
        qc = QuantumCircuit(n_qubits=4, n_layers=3)
        operators = qc.create_readout_operators(local_cost=False)
        
        # Should return a list
        assert isinstance(operators, list)
        assert len(operators) > 0
        
        # Global cost typically uses single operator
        assert len(operators) == 1
    
    def test_create_readout_operators_local(self):
        """Test local readout operator creation."""
        qc = QuantumCircuit(n_qubits=4, n_layers=3)
        operators = qc.create_readout_operators(local_cost=True)
        
        # Should return a list
        assert isinstance(operators, list)
        
        # Local cost should have one operator per qubit
        assert len(operators) == 4
    
    def test_get_symbols(self):
        """Test symbol extraction."""
        qc = QuantumCircuit(n_qubits=4, n_layers=3)
        
        # Get data symbols
        data_symbols = qc.get_data_symbols()
        assert len(data_symbols) > 0
        assert all(isinstance(s, sympy.Symbol) for s in data_symbols)
        
        # Get variational symbols
        var_symbols = qc.get_variational_symbols()
        assert len(var_symbols) > 0
        assert all(isinstance(s, sympy.Symbol) for s in var_symbols)
        
        # Should be different sets
        assert set(data_symbols).isdisjoint(set(var_symbols))
    
    def test_different_qubit_counts(self):
        """Test circuits with different qubit counts."""
        for n_qubits in [2, 4, 6, 8]:
            qc = QuantumCircuit(n_qubits=n_qubits, n_layers=2)
            circuit = qc.get_circuit()
            
            assert isinstance(circuit, cirq.Circuit)
            assert qc.n_qubits == n_qubits
            assert len(qc.qubits) == n_qubits
    
    def test_different_layer_counts(self):
        """Test circuits with different layer counts."""
        for n_layers in [2, 4, 6, 8]:
            qc = QuantumCircuit(n_qubits=4, n_layers=n_layers)
            circuit = qc.get_circuit()
            
            assert isinstance(circuit, cirq.Circuit)
            assert qc.n_layers == n_layers
    
    def test_circuit_parameters_count(self):
        """Test correct number of variational parameters."""
        n_qubits = 4
        n_layers = 3
        qc = QuantumCircuit(n_qubits=n_qubits, n_layers=n_layers)
        
        var_symbols = qc.get_variational_symbols()
        
        # Hardware-efficient ansatz: 2 params per qubit per layer (RY + RZ)
        expected_params = 2 * n_qubits * n_layers
        assert len(var_symbols) == expected_params


class TestCircuitExecution:
    """Test circuit execution and simulation."""
    
    def test_circuit_simulation(self):
        """Test that circuits can be simulated."""
        qc = QuantumCircuit(n_qubits=4, n_layers=2)
        circuit = qc.get_circuit()
        
        # Resolve symbols with random values
        data_symbols = qc.get_data_symbols()
        var_symbols = qc.get_variational_symbols()
        
        resolver = cirq.ParamResolver({
            **{s: np.random.rand() for s in data_symbols},
            **{s: np.random.rand() for s in var_symbols}
        })
        
        resolved_circuit = cirq.resolve_parameters(circuit, resolver)
        
        # Simulate
        simulator = cirq.Simulator()
        result = simulator.simulate(resolved_circuit)
        
        # Should produce a valid state vector
        assert result.final_state_vector is not None
        assert len(result.final_state_vector) == 2**4  # 4 qubits
    
    def test_expectation_value(self):
        """Test expectation value computation."""
        qc = QuantumCircuit(n_qubits=4, n_layers=2)
        circuit = qc.get_circuit()
        
        # Resolve with random parameters
        data_symbols = qc.get_data_symbols()
        var_symbols = qc.get_variational_symbols()
        resolver = cirq.ParamResolver({
            **{s: np.random.rand() for s in data_symbols},
            **{s: np.random.rand() for s in var_symbols}
        })
        
        resolved_circuit = cirq.resolve_parameters(circuit, resolver)
        
        # Compute expectation of Z on first qubit
        observable = cirq.Z(qc.qubits[0])
        
        simulator = cirq.Simulator()
        result = simulator.simulate(resolved_circuit)
        expectation = observable.expectation_from_state_vector(
            result.final_state_vector, 
            qubit_map={qc.qubits[0]: 0}
        )
        
        # Expectation value should be between -1 and 1
        assert -1 <= expectation.real <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
