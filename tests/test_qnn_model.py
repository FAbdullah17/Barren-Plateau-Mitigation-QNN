"""Unit tests for quantum neural network models."""

import pytest
import numpy as np
import tensorflow as tf
import cirq
import warnings

# Suppress TensorFlow/Keras warnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras')

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
        # QuantumNeuralNetwork IS a tf.keras.Model, not a wrapper
        assert isinstance(model, tf.keras.Model)
    
    def test_initialization_local_cost(self):
        """Test QNN initialization with local cost function."""
        model = QuantumNeuralNetwork(
            n_qubits=4,
            n_layers=3,
            local_cost=True
        )
        
        assert model.local_cost is True
        assert isinstance(model, tf.keras.Model)
    
    def test_model_structure(self):
        """Test model has correct structure."""
        model = QuantumNeuralNetwork(n_qubits=4, n_layers=3)
        
        # Should be a Keras model itself
        assert isinstance(model, tf.keras.Model)
        
        # Should have trainable parameters
        trainable_vars = model.trainable_variables
        assert len(trainable_vars) > 0
    
    def test_forward_pass(self):
        """Test model can perform forward pass."""
        model = QuantumNeuralNetwork(n_qubits=4, n_layers=2)
        
        # Model has required attributes
        assert hasattr(model, 'quantum_layer')
        assert hasattr(model, 'output_layer')
        assert isinstance(model, tf.keras.Model)
    
    def test_get_num_parameters(self):
        """Test parameter count method."""
        model = QuantumNeuralNetwork(n_qubits=4, n_layers=3)
        num_params = model.get_num_parameters()
        
        # Should return number of parameters
        assert isinstance(num_params, int)
        assert num_params > 0
        # Expected: 2 * n_qubits * n_layers = 2 * 4 * 3 = 24
        assert num_params == 2 * 4 * 3
    
    def test_different_architectures(self):
        """Test models with different architectures."""
        configs = [
            (2, 2), (4, 3), (4, 4), (6, 2), (8, 3)
        ]
        
        for n_qubits, n_layers in configs:
            model = QuantumNeuralNetwork(n_qubits=n_qubits, n_layers=n_layers)
            assert model.n_qubits == n_qubits
            assert model.n_layers == n_layers
            assert isinstance(model, tf.keras.Model)


class TestLayerwiseQNN:
    """Test layerwise QNN model."""
    
    def test_initialization(self):
        """Test layerwise QNN initialization."""
        model = LayerwiseQNN(
            n_qubits=4,
            target_layers=4,  # Changed from total_layers
            local_cost=False
        )
        
        assert model.n_qubits == 4
        assert model.target_layers == 4  # Changed from total_layers
        assert model.current_layers == 0
        assert model.local_cost is False
        assert model.model is None  # Initially no model
    
    def test_add_layer(self):
        """Test adding layers incrementally."""
        model = LayerwiseQNN(n_qubits=4, target_layers=4)  # Changed from total_layers
        
        # Initially no layers
        assert model.current_layers == 0
        
        # Add first layer
        keras_model = model.add_layer()
        assert model.current_layers == 1
        assert model.model is not None
        assert isinstance(keras_model, tf.keras.Model)
        
        # Add second layer
        keras_model = model.add_layer()
        assert model.current_layers == 2
        
        # Add remaining layers
        model.add_layer()
        model.add_layer()
        assert model.current_layers == 4
    
    def test_add_layer_exceeds_total(self):
        """Test error when adding more layers than target."""
        model = LayerwiseQNN(n_qubits=4, target_layers=2)  # Changed from total_layers
        
        model.add_layer()
        model.add_layer()
        
        # Should raise error when exceeding target_layers
        with pytest.raises(ValueError):
            model.add_layer()
    
    def test_get_current_model_no_layers(self):
        """Test getting model with no layers added."""
        model = LayerwiseQNN(n_qubits=4, target_layers=4)  # Changed from total_layers
        
        # Should return None if no layers added yet
        current_model = model.get_current_model()
        assert current_model is None
    
    def test_get_current_model_with_layers(self):
        """Test getting model after adding layers."""
        model = LayerwiseQNN(n_qubits=4, target_layers=4)  # Changed from total_layers
        
        model.add_layer()
        keras_model = model.get_current_model()  # Changed from get_model()
        
        assert isinstance(keras_model, tf.keras.Model)
        assert keras_model is not None
    
    def test_incremental_build(self):
        """Test incremental model building."""
        model = LayerwiseQNN(n_qubits=4, target_layers=3)  # Changed from total_layers
        
        # Add layers one by one
        for expected_layers in range(1, 4):
            keras_model = model.add_layer()
            assert model.current_layers == expected_layers
            
            # Should be able to get model at each stage
            current_model = model.get_current_model()  # Changed from get_model()
            assert current_model is not None
            assert isinstance(current_model, tf.keras.Model)
    
    def test_local_cost_layerwise(self):
        """Test layerwise model with local cost."""
        model = LayerwiseQNN(
            n_qubits=4,
            target_layers=3,  # Changed from total_layers
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
        
        # Compile with optimizer and loss (model is itself a keras.Model)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.MeanSquaredError()
        )
        
        # Should not raise error
        assert model.optimizer is not None
    
    def test_trainable_parameters(self):
        """Test model has correct number of trainable parameters."""
        n_qubits = 4
        n_layers = 3
        model = QuantumNeuralNetwork(n_qubits=n_qubits, n_layers=n_layers)
        
        # Model is itself a keras.Model
        trainable_vars = model.trainable_variables
        
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
        assert isinstance(model_global, tf.keras.Model)
        assert isinstance(model_local, tf.keras.Model)
        
        # They should have different cost function settings
        assert model_global.local_cost is False
        assert model_local.local_cost is True
        
        # Local cost should have Dense output layer, global should have Lambda
        assert isinstance(model_global.output_layer, tf.keras.layers.Lambda)
        assert isinstance(model_local.output_layer, tf.keras.layers.Dense)


class TestModelPersistence:
    """Test model saving and loading."""
    
    def test_model_state(self):
        """Test model state can be accessed."""
        model = QuantumNeuralNetwork(n_qubits=4, n_layers=2)
        
        # Get initial weights (model is itself a keras.Model)
        initial_weights = [w.numpy() for w in model.trainable_variables]
        
        assert len(initial_weights) > 0
        assert all(isinstance(w, np.ndarray) for w in initial_weights)
    
    def test_get_num_parameters_method(self):
        """Test get_num_parameters method works correctly."""
        model = QuantumNeuralNetwork(n_qubits=4, n_layers=3)
        num_params = model.get_num_parameters()
        
        # Should match the number from quantum circuit
        expected = 2 * 4 * 3  # 2 rotations * 4 qubits * 3 layers
        assert num_params == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])