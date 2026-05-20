"""Unit tests for MNIST data loader."""

import pytest
import numpy as np
import tensorflow as tf
from src.data.mnist_loader import load_mnist_binary, encode_data_for_qnn


class TestMNISTLoader:
    """Test MNIST binary classification loader."""
    
    def test_load_mnist_binary_basic(self):
        """Test basic data loading with default parameters."""
        X_train, X_test, y_train, y_test = load_mnist_binary(
            digit1=3, digit2=6, train_size=100, test_size=20
        )
        
        # Check shapes
        assert X_train.shape == (100, 4), f"Expected (100, 4), got {X_train.shape}"
        assert X_test.shape == (20, 4), f"Expected (20, 4), got {X_test.shape}"
        assert y_train.shape == (100,), f"Expected (100,), got {y_train.shape}"
        assert y_test.shape == (20,), f"Expected (20,), got {y_test.shape}"
        
        # Check data types
        assert X_train.dtype == np.float32
        assert X_test.dtype == np.float32
        assert y_train.dtype == np.int32
        assert y_test.dtype == np.int32
        
        # Check value ranges
        assert np.all((X_train >= 0) & (X_train <= 1)), "Features not normalized"
        assert np.all((X_test >= 0) & (X_test <= 1)), "Features not normalized"
        assert set(y_train.tolist()) <= {0, 1}, "Labels not binary"
        assert set(y_test.tolist()) <= {0, 1}, "Labels not binary"
    
    def test_load_mnist_different_digits(self):
        """Test loading with different digit pairs."""
        X_train, X_test, y_train, y_test = load_mnist_binary(
            digit1=0, digit2=1, train_size=50, test_size=10
        )
        
        assert X_train.shape[0] == 50
        assert X_test.shape[0] == 10
        assert len(y_train) == 50
        assert len(y_test) == 10
    
    def test_load_mnist_image_size(self):
        """Test custom image size parameter."""
        # Default is 4 dominant features
        X_train, _, _, _ = load_mnist_binary(train_size=10, test_size=5)
        assert X_train.shape[1] == 4
        
        # Custom 8x8 = 64 features
        X_train_large, _, _, _ = load_mnist_binary(
            train_size=10, test_size=5, image_size=(8, 8)
        )
        assert X_train_large.shape[1] == 64
    
    def test_load_mnist_reproducibility(self):
        """Test that loading with same seed gives same data."""
        X1, _, y1, _ = load_mnist_binary(train_size=50, test_size=10, seed=42)
        X2, _, y2, _ = load_mnist_binary(train_size=50, test_size=10, seed=42)
        
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
    
    def test_load_mnist_different_seeds(self):
        """Test that different seeds give different data."""
        X1, _, y1, _ = load_mnist_binary(train_size=50, test_size=10, seed=42)
        X2, _, y2, _ = load_mnist_binary(train_size=50, test_size=10, seed=123)
        
        # Should not be identical (very high probability)
        assert not np.array_equal(X1, X2) or not np.array_equal(y1, y2)
    
    def test_invalid_digits(self):
        """Test error handling for invalid digit values."""
        with pytest.raises((ValueError, AssertionError)):
            load_mnist_binary(digit1=10, digit2=5)
        
        with pytest.raises((ValueError, AssertionError)):
            load_mnist_binary(digit1=3, digit2=3)  # Same digits


class TestEncodeData:
    """Test quantum encoding functions."""
    
    def test_encode_data_for_qnn_shape(self):
        """Test encoding produces correct shape."""
        # Create sample data
        X = np.random.rand(10, 4).astype(np.float32)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
        
        # Encode
        X_encoded, y_encoded = encode_data_for_qnn(X, y, n_qubits=4)
        
        # Check shapes - should preserve sample count
        assert len(X_encoded) == 10
        assert len(y_encoded) == 10
    
    def test_encode_data_for_qnn_labels(self):
        """Test label encoding to -1/+1."""
        X = np.random.rand(10, 4).astype(np.float32)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
        
        _, y_encoded = encode_data_for_qnn(X, y, n_qubits=4)
        
        # Should be converted to -1 and +1
        unique_labels = np.unique(y_encoded)
        assert set(unique_labels.tolist()) == {-1, 1}
    
    def test_encode_data_mismatch(self):
        """Test error handling for mismatched X and y lengths."""
        X = np.random.rand(10, 4).astype(np.float32)
        y = np.array([0, 1, 0, 1, 0], dtype=np.int32)  # Wrong length
        
        with pytest.raises((ValueError, AssertionError)):
            encode_data_for_qnn(X, y, n_qubits=4)
    
    def test_encode_preserves_data(self):
        """Test that encoding preserves original data integrity."""
        X = np.random.rand(5, 4).astype(np.float32)
        y = np.array([0, 1, 0, 1, 0], dtype=np.int32)
        
        X_before = X.copy()
        y_before = y.copy()
        
        encode_data_for_qnn(X, y, n_qubits=4)
        
        # Original arrays should be unchanged
        np.testing.assert_array_equal(X, X_before)
        np.testing.assert_array_equal(y, y_before)


class TestDataIntegration:
    """Integration tests for complete data pipeline."""
    
    def test_full_pipeline(self):
        """Test complete pipeline from loading to encoding."""
        # Load data
        X_train, X_test, y_train, y_test = load_mnist_binary(
            digit1=3, digit2=6, train_size=100, test_size=20
        )
        
        # Encode for QNN
        X_train_encoded, y_train_encoded = encode_data_for_qnn(
            X_train, y_train, n_qubits=4
        )
        X_test_encoded, y_test_encoded = encode_data_for_qnn(
            X_test, y_test, n_qubits=4
        )
        
        # Verify pipeline output
        assert len(X_train_encoded) == 100
        assert len(X_test_encoded) == 20
        assert set(np.unique(y_train_encoded).tolist()) == {-1, 1}
        assert set(np.unique(y_test_encoded).tolist()) == {-1, 1}
    
    def test_batch_processing(self):
        """Test data can be batched correctly."""
        X_train, _, y_train, _ = load_mnist_binary(train_size=100, test_size=20)
        X_encoded, y_encoded = encode_data_for_qnn(X_train, y_train, n_qubits=4)
        
        # Create TensorFlow dataset
        batch_size = 20
        dataset = tf.data.Dataset.from_tensor_slices((X_encoded, y_encoded))
        dataset = dataset.batch(batch_size)
        
        # Check batching works
        for batch_x, batch_y in dataset.take(1):
            assert batch_x.shape[0] == batch_size
            assert batch_y.shape[0] == batch_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
