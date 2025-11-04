"""Data loading and preprocessing utilities for MNIST binary classification.

Developer Assignment (Weeks 1-2):
    Primary: Fahad Abdullah - Data pipeline & quantum encoding
    Review: Asma Zubair - Validation framework
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional


def load_mnist_binary(
    digit1: int = 3,
    digit2: int = 6,
    train_size: int = 1000,
    test_size: int = 200,
    image_size: Tuple[int, int] = (4, 4),
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess MNIST data for binary classification.
    
    Args:
        digit1: First digit class (default: 3)
        digit2: Second digit class (default: 6)
        train_size: Number of training samples (default: 1000)
        test_size: Number of test samples (default: 200)
        image_size: Target image dimensions (default: (4, 4))
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
        - X: Flattened images normalized to [0, 1]
        - y: Binary labels (0 or 1)
    """
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    # Load MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Filter for binary classification
    train_filter = np.isin(y_train, [digit1, digit2])
    test_filter = np.isin(y_test, [digit1, digit2])
    
    x_train = x_train[train_filter]
    y_train = y_train[train_filter]
    x_test = x_test[test_filter]
    y_test = y_test[test_filter]
    
    # Convert labels to binary (0, 1)
    y_train = (y_train == digit2).astype(int)
    y_test = (y_test == digit2).astype(int)
    
    # Downsample images using bilinear interpolation
    x_train = tf.image.resize(
        x_train[..., np.newaxis], image_size, method='bilinear'
    ).numpy().squeeze()
    x_test = tf.image.resize(
        x_test[..., np.newaxis], image_size, method='bilinear'
    ).numpy().squeeze()
    
    # Normalize to [0, 1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Flatten images
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)
    
    # Sample specified number of examples
    if len(x_train) > train_size:
        train_indices = np.random.choice(len(x_train), train_size, replace=False)
        x_train = x_train[train_indices]
        y_train = y_train[train_indices]
    
    if len(x_test) > test_size:
        test_indices = np.random.choice(len(x_test), test_size, replace=False)
        x_test = x_test[test_indices]
        y_test = y_test[test_indices]
    
    return x_train, y_train, x_test, y_test


def encode_data_for_qnn(data: np.ndarray) -> np.ndarray:
    """
    Encode classical data for quantum neural network input.
    
    Args:
        data: Flattened image data, shape (n_samples, n_features)
        
    Returns:
        Encoded data scaled to rotation angles
    """
    # Scale to rotation angles [0, π]
    return data * np.pi


if __name__ == "__main__":
    # Test data loading
    X_train, y_train, X_test, y_test = load_mnist_binary(seed=42)
    print(f"Training set: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Test set: {X_test.shape}, Labels: {y_test.shape}")
    print(f"Label distribution (train): {np.bincount(y_train)}")
    print(f"Label distribution (test): {np.bincount(y_test)}")
