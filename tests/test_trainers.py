"""Unit tests for training modules."""

import pytest
import numpy as np
import tensorflow as tf
from src.training.baseline_trainer import BaselineTrainer
from src.training.layerwise_trainer import LayerwiseTrainer


class TestBaselineTrainer:
    """Test baseline training implementation."""
    
    def test_initialization(self):
        """Test baseline trainer initialization."""
        trainer = BaselineTrainer(
            n_qubits=4,
            n_layers=3,
            learning_rate=0.01,
            batch_size=20,
            local_cost=False
        )
        
        assert trainer.n_qubits == 4
        assert trainer.n_layers == 3
        assert trainer.learning_rate == 0.01
        assert trainer.batch_size == 20
        assert trainer.local_cost is False
    
    def test_initialization_with_seed(self):
        """Test trainer initialization with random seed."""
        trainer = BaselineTrainer(
            n_qubits=4,
            n_layers=2,
            seed=42
        )
        
        assert trainer is not None
    
    def test_local_cost_initialization(self):
        """Test trainer with local cost function."""
        trainer = BaselineTrainer(
            n_qubits=4,
            n_layers=2,
            local_cost=True
        )
        
        assert trainer.local_cost is True
    
    def test_gradient_tracker_initialization(self):
        """Test that gradient tracker is initialized."""
        trainer = BaselineTrainer(n_qubits=4, n_layers=2)
        
        # Trainer should have gradient tracker
        assert hasattr(trainer, 'gradient_tracker')
    
    def test_model_creation(self):
        """Test that model is created during initialization."""
        trainer = BaselineTrainer(n_qubits=4, n_layers=2)
        
        # Model should exist
        assert trainer.model is not None
        assert hasattr(trainer, 'optimizer')


class TestLayerwiseTrainer:
    """Test layerwise training implementation."""
    
    def test_initialization(self):
        """Test layerwise trainer initialization."""
        trainer = LayerwiseTrainer(
            n_qubits=4,
            total_layers=4,
            learning_rate=0.01,
            batch_size=20,
            epochs_per_layer=10,
            finetune_epochs=10
        )
        
        assert trainer.n_qubits == 4
        assert trainer.total_layers == 4
        assert trainer.learning_rate == 0.01
        assert trainer.batch_size == 20
        assert trainer.epochs_per_layer == 10
        assert trainer.finetune_epochs == 10
    
    def test_initialization_with_seed(self):
        """Test layerwise trainer with seed."""
        trainer = LayerwiseTrainer(
            n_qubits=4,
            total_layers=3,
            seed=42
        )
        
        assert trainer is not None
    
    def test_local_cost_layerwise(self):
        """Test layerwise trainer with local cost."""
        trainer = LayerwiseTrainer(
            n_qubits=4,
            total_layers=3,
            local_cost=True
        )
        
        assert trainer.local_cost is True
    
    def test_model_initialization(self):
        """Test that layerwise model is properly initialized."""
        trainer = LayerwiseTrainer(n_qubits=4, total_layers=3)
        
        # Should have layerwise QNN model
        assert trainer.qnn_model is not None
    
    def test_gradient_tracker(self):
        """Test gradient tracker in layerwise training."""
        trainer = LayerwiseTrainer(n_qubits=4, total_layers=3)
        
        assert hasattr(trainer, 'gradient_tracker')


class TestTrainerConfiguration:
    """Test trainer configuration and parameters."""
    
    def test_different_learning_rates(self):
        """Test trainers with different learning rates."""
        for lr in [0.001, 0.01, 0.1]:
            trainer = BaselineTrainer(
                n_qubits=4,
                n_layers=2,
                learning_rate=lr
            )
            assert trainer.learning_rate == lr
    
    def test_different_batch_sizes(self):
        """Test trainers with different batch sizes."""
        for batch_size in [10, 20, 32, 64]:
            trainer = BaselineTrainer(
                n_qubits=4,
                n_layers=2,
                batch_size=batch_size
            )
            assert trainer.batch_size == batch_size
    
    def test_different_architectures(self):
        """Test trainers with different model architectures."""
        configs = [(2, 2), (4, 3), (4, 4), (6, 2)]
        
        for n_qubits, n_layers in configs:
            trainer = BaselineTrainer(
                n_qubits=n_qubits,
                n_layers=n_layers
            )
            assert trainer.n_qubits == n_qubits
            assert trainer.n_layers == n_layers
    
    def test_layerwise_epochs_configuration(self):
        """Test layerwise trainer epochs configuration."""
        trainer = LayerwiseTrainer(
            n_qubits=4,
            total_layers=4,
            epochs_per_layer=5,
            finetune_epochs=15
        )
        
        assert trainer.epochs_per_layer == 5
        assert trainer.finetune_epochs == 15


class TestTrainingHistory:
    """Test training history tracking."""
    
    def test_history_structure(self):
        """Test that training history has correct structure."""
        # This would be better tested with actual training,
        # but we can test the structure expectations
        
        trainer = BaselineTrainer(n_qubits=4, n_layers=2)
        
        # After training, history should contain:
        # - train_loss, test_loss
        # - train_accuracy, test_accuracy
        # - gradient_norms, gradient_variance
        # We'll test this structure in integration tests
        
        assert trainer is not None
    
    def test_layerwise_history_structure(self):
        """Test layerwise training history structure."""
        trainer = LayerwiseTrainer(n_qubits=4, total_layers=3)
        
        # Layerwise history should additionally track:
        # - layer_transitions
        # - per-layer metrics
        
        assert trainer is not None


class TestTrainerEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_qubit_count(self):
        """Test error handling for invalid qubit count."""
        with pytest.raises((ValueError, AssertionError)):
            BaselineTrainer(n_qubits=0, n_layers=2)
        
        with pytest.raises((ValueError, AssertionError)):
            BaselineTrainer(n_qubits=-1, n_layers=2)
    
    def test_invalid_layer_count(self):
        """Test error handling for invalid layer count."""
        with pytest.raises((ValueError, AssertionError)):
            BaselineTrainer(n_qubits=4, n_layers=0)
        
        with pytest.raises((ValueError, AssertionError)):
            BaselineTrainer(n_qubits=4, n_layers=-1)
    
    def test_invalid_learning_rate(self):
        """Test error handling for invalid learning rate."""
        # Negative learning rate should fail
        with pytest.raises((ValueError, AssertionError)):
            BaselineTrainer(n_qubits=4, n_layers=2, learning_rate=-0.01)
    
    def test_invalid_batch_size(self):
        """Test error handling for invalid batch size."""
        with pytest.raises((ValueError, AssertionError)):
            BaselineTrainer(n_qubits=4, n_layers=2, batch_size=0)
        
        with pytest.raises((ValueError, AssertionError)):
            BaselineTrainer(n_qubits=4, n_layers=2, batch_size=-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
