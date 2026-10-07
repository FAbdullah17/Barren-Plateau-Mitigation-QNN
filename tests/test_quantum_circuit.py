"""Tests for hardware-efficient ansatz circuit construction.

Covers the actual ``QuantumCircuit`` API (readout-operator tests live in
``test_cost_functions.py``).
"""

import pytest
import cirq
import sympy

from src.models.quantum_circuit import QuantumCircuit


class TestQuantumCircuit:
    def test_initialization(self):
        qc = QuantumCircuit(n_qubits=4, n_layers=3)
        assert qc.n_qubits == 4
        assert qc.n_layers == 3
        assert len(qc.qubits) == 4
        assert all(isinstance(q, cirq.GridQubit) for q in qc.qubits)

    @pytest.mark.parametrize("n_qubits", [4, 6, 8])
    @pytest.mark.parametrize("n_layers", [2, 4, 6])
    def test_parameter_count(self, n_qubits, n_layers):
        qc = QuantumCircuit(n_qubits=n_qubits, n_layers=n_layers)
        params = qc.get_parameters()
        assert len(params) == 2 * n_qubits * n_layers
        assert all(isinstance(p, sympy.Symbol) for p in params)

    def test_parameter_names_unique(self):
        qc = QuantumCircuit(n_qubits=4, n_layers=3)
        names = [str(p) for p in qc.get_parameters()]
        assert len(names) == len(set(names))

    def test_layer_parameters(self):
        qc = QuantumCircuit(n_qubits=4, n_layers=4)
        layer = qc.get_layer_parameters(2)
        assert len(layer) == 8

    def test_layer_parameters_out_of_range(self):
        qc = QuantumCircuit(n_qubits=4, n_layers=2)
        with pytest.raises(ValueError):
            qc.get_layer_parameters(5)

    def test_get_circuit(self):
        qc = QuantumCircuit(n_qubits=4, n_layers=3)
        circuit = qc.get_circuit()
        assert isinstance(circuit, cirq.Circuit)
        assert len(list(circuit.all_operations())) > 0

    def test_circuit_up_to_layer_grows(self):
        qc = QuantumCircuit(n_qubits=4, n_layers=4)
        c1 = qc.get_circuit_up_to_layer(0)
        c2 = qc.get_circuit_up_to_layer(1)
        assert len(list(c2.all_operations())) >= len(list(c1.all_operations()))

    def test_circuit_up_to_layer_invalid(self):
        qc = QuantumCircuit(n_qubits=4, n_layers=3)
        with pytest.raises((ValueError, IndexError)):
            qc.get_circuit_up_to_layer(10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
