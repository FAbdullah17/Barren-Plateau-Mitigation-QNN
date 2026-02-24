"""Data loading and preprocessing utilities for MNIST binary classification.

Handles downloading, filtering, downsampling, and encoding of MNIST digit
images for use in quantum neural network experiments. Supports binary
classification between any two digit classes with configurable image
dimensions and dataset sizes.
"""

import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging (0=all, 3=errors only)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional
from pathlib import Path

# Disable TensorFlow logging
tf.get_logger().setLevel('ERROR')


def load_mnist_binary(
    digit1: int = 3,
    digit2: int = 6,
    train_size: int = 1000,
    test_size: int = 200,
    image_size: Tuple[int, int] = (4, 4),
    seed: Optional[int] = None,
    save_filtered: bool = False,
    filtered_filename: str = 'mnist_3_6.npz'
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
    
    # Set up data directory in project root
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data' / 'raw'
    data_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load MNIST (will cache to project data directory)
    mnist_path = data_dir / 'mnist.npz'
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path=str(mnist_path))
    
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
    
    # Optionally save the filtered/processed dataset as compressed .npz
    if save_filtered:
        filtered_path = processed_dir / filtered_filename
        np.savez_compressed(
            filtered_path,
            X_train=x_train,
            y_train=y_train,
            X_test=x_test,
            y_test=y_test
        )
        print(f"✓ Filtered dataset saved to: {filtered_path}")

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
    # Test data loading and save filtered dataset locally
    X_train, y_train, X_test, y_test = load_mnist_binary(seed=42, save_filtered=True)
    print(f"Training set: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Test set: {X_test.shape}, Labels: {y_test.shape}")
    print(f"Label distribution (train): {np.bincount(y_train)}")
    print(f"Label distribution (test): {np.bincount(y_test)}")
