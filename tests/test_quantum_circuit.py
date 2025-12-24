"""Unit tests for quantum circuit construction."""

import pytest
import cirq
import sympy
import numpy as np
from src.models.quantum_circuit import QuantumCircuit, create_readout_operators


class TestQuantumCircuit:
    """Test quantum circuit builder."""
    
    def test_initialization(self):
        """Test circuit initialization."""
        qc = QuantumCircuit(n_qubits=4, n_layers=3)
        
        assert qc.n_qubits == 4
        assert qc.n_layers == 3
        assert len(qc.qubits) == 4
        assert all(isinstance(q, cirq.GridQubit) for q in qc.qubits)
    
    def test_get_circuit(self):
        """Test full circuit retrieval."""
        qc = QuantumCircuit(n_qubits=4, n_layers=3)
        full_circuit = qc.get_circuit()
        
        assert isinstance(full_circuit, cirq.Circuit)
        assert len(list(full_circuit.all_operations())) > 0
    
    def test_get_circuit_up_to_layer(self):
        """Test layerwise circuit construction."""
        qc = QuantumCircuit(n_qubits=4, n_layers=4)
        
        # Get circuits up to different layers (0-indexed: 0, 1, 2, 3)
        circuit_0 = qc.get_circuit_up_to_layer(0)  # Layer 0 only
        circuit_1 = qc.get_circuit_up_to_layer(1)  # Layers 0-1
        circuit_2 = qc.get_circuit_up_to_layer(2)  # Layers 0-2
        circuit_3 = qc.get_circuit_up_to_layer(3)  # Layers 0-3 (all 4 layers)
        
        # Check they're all valid
        assert isinstance(circuit_0, cirq.Circuit)
        assert isinstance(circuit_1, cirq.Circuit)
        assert isinstance(circuit_2, cirq.Circuit)
        assert isinstance(circuit_3, cirq.Circuit)
        
        # Each subsequent circuit should have more operations
        ops_0 = len(list(circuit_0.all_operations()))
        ops_1 = len(list(circuit_1.all_operations()))
        ops_2 = len(list(circuit_2.all_operations()))
        ops_3 = len(list(circuit_3.all_operations()))
        
        assert ops_1 >= ops_0
        assert ops_2 >= ops_1
        assert ops_3 >= ops_2
    
    def test_get_circuit_up_to_layer_invalid(self):
        """Test error handling for invalid layer numbers."""
        qc = QuantumCircuit(n_qubits=4, n_layers=3)
        
        # Test index >= n_layers (0-indexed: valid are 0, 1, 2)
        with pytest.raises((ValueError, IndexError)):
            qc.get_circuit_up_to_layer(3)  # Should fail (only 0, 1, 2 are valid)
    
    def test_get_parameters(self):
        """Test getting all parameters as flat list."""
        qc = QuantumCircuit(n_qubits=4, n_layers=3)
        all_params = qc.get_parameters()
        
        # Should return a list of sympy symbols
        assert isinstance(all_params, list)
        assert len(all_params) > 0
        assert all(isinstance(p, sympy.Symbol) for p in all_params)
        
        # Hardware-efficient ansatz: 2 params per qubit per layer (RY + RZ)
        expected_params = 2 * 4 * 3  # 2 * n_qubits * n_layers
        assert len(all_params) == expected_params
    
    def test_get_layer_parameters(self):
        """Test getting parameters for a specific layer."""
        qc = QuantumCircuit(n_qubits=4, n_layers=3)
        
        # Get parameters for each layer (0-indexed)
        for layer_idx in range(3):
            layer_params = qc.get_layer_parameters(layer_idx)
            
            assert isinstance(layer_params, list)
            assert len(layer_params) > 0
            assert all(isinstance(p, sympy.Symbol) for p in layer_params)
            
            # Each layer should have 8 parameters (2 rotations × 4 qubits)
            expected_params_per_layer = 2 * 4
            assert len(layer_params) == expected_params_per_layer
    
    def test_parameters_per_layer(self):
        """
        Week 1 Task (Frahan): Verify 8 parameters per layer.
        
        Each layer should have exactly: 2 rotations (RY, RZ) × 4 qubits = 8 parameters
        Tests at experimental depths: 4, 6, 8 layers
        """
        n_qubits = 4  # Fixed for this project
        
        # Test for depths 4, 6, 8 (our experimental depths)
        for n_layers in [4, 6, 8]:
            qc = QuantumCircuit(n_qubits=n_qubits, n_layers=n_layers)
            
            # Get parameters for each layer (0-indexed: 0, 1, ..., n_layers-1)
            for layer_idx in range(n_layers):
                layer_params = qc.get_layer_parameters(layer_idx)
                
                # Each layer should have: 2 rotations (RY, RZ) × 4 qubits = 8 parameters
                expected_params_per_layer = 2 * n_qubits
                assert len(layer_params) == expected_params_per_layer, \
                    f"Layer {layer_idx} should have {expected_params_per_layer} parameters, " \
                    f"got {len(layer_params)}"
                
                # Verify parameter naming pattern (RY and RZ for each qubit)
                # Should have symbols like: theta_0_0_ry, theta_0_0_rz, theta_0_1_ry, etc.
                for qubit_idx in range(n_qubits):
                    ry_symbol = layer_params[2 * qubit_idx]
                    rz_symbol = layer_params[2 * qubit_idx + 1]
                    
                    assert f'theta_{layer_idx}_{qubit_idx}_ry' in str(ry_symbol), \
                        f"Expected RY parameter for layer {layer_idx}, qubit {qubit_idx}"
                    assert f'theta_{layer_idx}_{qubit_idx}_rz' in str(rz_symbol), \
                        f"Expected RZ parameter for layer {layer_idx}, qubit {qubit_idx}"
        
        print("\n✓ Verified: Each layer has exactly 8 parameters (2 × 4 qubits)")
    
    def test_circuit_parameters_count(self):
        """Test correct number of variational parameters (total)."""
        n_qubits = 4
        n_layers = 3
        qc = QuantumCircuit(n_qubits=n_qubits, n_layers=n_layers)
        
        # Use the actual method name: get_parameters()
        all_params = qc.get_parameters()
        
        # Hardware-efficient ansatz: 2 params per qubit per layer (RY + RZ)
        expected_params = 2 * n_qubits * n_layers
        assert len(all_params) == expected_params, \
            f"Expected {expected_params} total parameters, got {len(all_params)}"
    
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
    
    def test_visualize(self):
        """Test circuit visualization."""
        qc = QuantumCircuit(n_qubits=4, n_layers=2)
        visualization = qc.visualize()
        
        assert isinstance(visualization, str)
        assert len(visualization) > 0


class TestReadoutOperators:
    """Test readout operator creation."""
    
    def test_create_readout_operators_global(self):
        """Test global readout operator creation."""
        operators = create_readout_operators(n_qubits=4, local=False)
        
        # Should return a list
        assert isinstance(operators, list)
        assert len(operators) > 0
        
        # Global cost typically uses single operator (first qubit)
        assert len(operators) == 1
    
    def test_create_readout_operators_local(self):
        """Test local readout operator creation."""
        operators = create_readout_operators(n_qubits=4, local=True)
        
        # Should return a list
        assert isinstance(operators, list)
        
        # Local cost should have one operator per qubit
        assert len(operators) == 4
    
    def test_readout_operators_different_qubits(self):
        """Test readout operators with different qubit counts."""
        for n_qubits in [2, 4, 6, 8]:
            # Global
            global_ops = create_readout_operators(n_qubits=n_qubits, local=False)
            assert len(global_ops) == 1
            
            # Local
            local_ops = create_readout_operators(n_qubits=n_qubits, local=True)
            assert len(local_ops) == n_qubits


class TestCircuitExecution:
    """Test circuit execution and simulation."""
    
    def test_circuit_simulation(self):
        """Test that circuits can be simulated."""
        qc = QuantumCircuit(n_qubits=4, n_layers=2)
        circuit = qc.get_circuit()
        
        # Get all parameters
        all_params = qc.get_parameters()
        
        # Create resolver with random values for all parameters
        resolver = cirq.ParamResolver({
            str(param): np.random.rand() * 2 * np.pi  # Random angle in [0, 2π]
            for param in all_params
        })
        
        resolved_circuit = cirq.resolve_parameters(circuit, resolver)
        
        # Simulate
        simulator = cirq.Simulator()
        result = simulator.simulate(resolved_circuit)
        
        # Should produce a valid state vector
        assert result.final_state_vector is not None
        assert len(result.final_state_vector) == 2**4  # 4 qubits = 16 states
    
    def test_expectation_value(self):
        """Test expectation value computation."""
        qc = QuantumCircuit(n_qubits=4, n_layers=2)
        circuit = qc.get_circuit()
        
        # Get all parameters
        all_params = qc.get_parameters()
        
        # Create resolver with random values
        resolver = cirq.ParamResolver({
            str(param): np.random.rand() * 2 * np.pi
            for param in all_params
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