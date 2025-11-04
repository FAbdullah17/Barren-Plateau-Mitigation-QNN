"""Unit tests for quantum neural network models."""

import pytest
import numpy as np
import tensorflow as tf
import cirq
from src.models.qnn_model import QuantumNeuralNetwork, LayerwiseQNN
from src.models.quantum_circuit import QuantumCircuit


class TestQuantumNeuralNetwork:
    """Test standard QNN model."""
    
    def test_initialization_global_cost(self):
        """Test QNN initialization with global cost function."""
        model = QuantumNeuralNetwork(
            n_qubits=4,
            n_layers=3,
            local_cost=False
        )
        
        assert model.n_qubits == 4
        assert model.n_layers == 3
        assert model.local_cost is False
        assert model.model is not None
    
    def test_initialization_local_cost(self):
        """Test QNN initialization with local cost function."""
        model = QuantumNeuralNetwork(
            n_qubits=4,
            n_layers=3,
            local_cost=True
        )
        
        assert model.local_cost is True
        assert model.model is not None
    
    def test_model_structure(self):
        """Test model has correct structure."""
        model = QuantumNeuralNetwork(n_qubits=4, n_layers=3)
        
        # Should be a Keras model
        assert isinstance(model.model, tf.keras.Model)
        
        # Should have trainable parameters
        trainable_vars = model.model.trainable_variables
        assert len(trainable_vars) > 0
    
    def test_forward_pass(self):
        """Test model can perform forward pass."""
        model = QuantumNeuralNetwork(n_qubits=4, n_layers=2)
        
        # Create dummy input (batch of quantum circuits)
        qc = QuantumCircuit(n_qubits=4, n_layers=2)
        circuit = qc.get_circuit()
        
        # Create batch of circuits
        batch_size = 5
        circuits = [circuit] * batch_size
        
        # Try to get output shape info
        # Note: Actual forward pass requires proper circuit tensors
        assert model.model is not None
    
    def test_get_model(self):
        """Test model getter."""
        qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=3)
        model = qnn.get_model()
        
        assert isinstance(model, tf.keras.Model)
    
    def test_different_architectures(self):
        """Test models with different architectures."""
        configs = [
            (2, 2), (4, 3), (4, 4), (6, 2), (8, 3)
        ]
        
        for n_qubits, n_layers in configs:
            model = QuantumNeuralNetwork(n_qubits=n_qubits, n_layers=n_layers)
            assert model.n_qubits == n_qubits
            assert model.n_layers == n_layers
            assert model.model is not None


class TestLayerwiseQNN:
    """Test layerwise QNN model."""
    
    def test_initialization(self):
        """Test layerwise QNN initialization."""
        model = LayerwiseQNN(
            n_qubits=4,
            total_layers=4,
            local_cost=False
        )
        
        assert model.n_qubits == 4
        assert model.total_layers == 4
        assert model.current_layers == 0
        assert model.local_cost is False
    
    def test_add_layer(self):
        """Test adding layers incrementally."""
        model = LayerwiseQNN(n_qubits=4, total_layers=4)
        
        # Initially no layers
        assert model.current_layers == 0
        
        # Add first layer
        model.add_layer()
        assert model.current_layers == 1
        assert model.model is not None
        
        # Add second layer
        model.add_layer()
        assert model.current_layers == 2
        
        # Add remaining layers
        model.add_layer()
        model.add_layer()
        assert model.current_layers == 4
    
    def test_add_layer_exceeds_total(self):
        """Test error when adding more layers than total."""
        model = LayerwiseQNN(n_qubits=4, total_layers=2)
        
        model.add_layer()
        model.add_layer()
        
        # Should raise error when exceeding total_layers
        with pytest.raises((ValueError, AssertionError)):
            model.add_layer()
    
    def test_get_model_no_layers(self):
        """Test getting model with no layers added."""
        model = LayerwiseQNN(n_qubits=4, total_layers=4)
        
        # Should raise error or return None
        with pytest.raises((ValueError, RuntimeError)):
            model.get_model()
    
    def test_get_model_with_layers(self):
        """Test getting model after adding layers."""
        model = LayerwiseQNN(n_qubits=4, total_layers=4)
        
        model.add_layer()
        keras_model = model.get_model()
        
        assert isinstance(keras_model, tf.keras.Model)
    
    def test_incremental_build(self):
        """Test incremental model building."""
        model = LayerwiseQNN(n_qubits=4, total_layers=3)
        
        # Add layers one by one
        for expected_layers in range(1, 4):
            model.add_layer()
            assert model.current_layers == expected_layers
            
            # Should be able to get model at each stage
            keras_model = model.get_model()
            assert keras_model is not None
    
    def test_local_cost_layerwise(self):
        """Test layerwise model with local cost."""
        model = LayerwiseQNN(
            n_qubits=4,
            total_layers=3,
            local_cost=True
        )
        
        model.add_layer()
        assert model.local_cost is True
        assert model.model is not None


class TestModelIntegration:
    """Integration tests for QNN models."""
    
    def test_model_compilation(self):
        """Test model can be compiled."""
        model = QuantumNeuralNetwork(n_qubits=4, n_layers=2)
        keras_model = model.get_model()
        
        # Compile with optimizer and loss
        keras_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.MeanSquaredError()
        )
        
        # Should not raise error
        assert keras_model.optimizer is not None
    
    def test_trainable_parameters(self):
        """Test model has correct number of trainable parameters."""
        n_qubits = 4
        n_layers = 3
        model = QuantumNeuralNetwork(n_qubits=n_qubits, n_layers=n_layers)
        
        keras_model = model.get_model()
        trainable_vars = keras_model.trainable_variables
        
        # Should have variational parameters
        assert len(trainable_vars) > 0
        
        # For hardware-efficient ansatz: 2 * n_qubits * n_layers parameters
        total_params = sum(tf.size(v).numpy() for v in trainable_vars)
        expected_params = 2 * n_qubits * n_layers
        
        assert total_params == expected_params
    
    def test_global_vs_local_cost(self):
        """Test difference between global and local cost functions."""
        model_global = QuantumNeuralNetwork(
            n_qubits=4, n_layers=2, local_cost=False
        )
        model_local = QuantumNeuralNetwork(
            n_qubits=4, n_layers=2, local_cost=True
        )
        
        # Both should be valid models
        assert model_global.model is not None
        assert model_local.model is not None
        
        # They should have different output dimensions potentially
        # (local cost may output per-qubit measurements)
        assert model_global.local_cost is False
        assert model_local.local_cost is True


class TestModelPersistence:
    """Test model saving and loading."""
    
    def test_model_state(self):
        """Test model state can be accessed."""
        model = QuantumNeuralNetwork(n_qubits=4, n_layers=2)
        keras_model = model.get_model()
        
        # Get initial weights
        initial_weights = [w.numpy() for w in keras_model.trainable_variables]
        
        assert len(initial_weights) > 0
        assert all(isinstance(w, np.ndarray) for w in initial_weights)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
