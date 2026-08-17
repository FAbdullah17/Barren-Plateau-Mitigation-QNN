"""Data loading and preprocessing for MNIST binary classification.

Loads MNIST, filters to a binary digit pair, subsamples with a dedicated
data seed, then reduces to a fixed low-dimensional representation:

    28x28 -> 4x4 bilinear downsample -> 16 features -> PCA -> n_components

The PCA is fitted on the training split only (no leakage) and every one of
the ``n_components`` features is consumed by the circuit encoder. This is
intentionally a *fixed low-dimensional classical input pipeline*, not a
natural high-dimensional MNIST representation.

Seed policy: ``data_seed`` governs subsampling only, via a private
``np.random.default_rng``. This module never touches the global NumPy seed
state (parameter initialization and training randomness are controlled by
separate seeds elsewhere). It is also intentionally free of TensorFlow
dependencies so that the data pipeline can be built and tested in isolation.
"""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import urllib.request
import numpy as np
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
from typing import Tuple, Optional, Dict
from pathlib import Path

MNIST_FLAT_DIM = 28 * 28
DOWNSAMPLED_FEATURES = 16  # 4x4 downsample

MNIST_DOWNLOAD_URL = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
)


def _load_mnist_raw(mnist_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the raw MNIST npz from disk, downloading it if necessary."""
    if not mnist_path.exists():
        print(f"Downloading MNIST from {MNIST_DOWNLOAD_URL} ...")
        mnist_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MNIST_DOWNLOAD_URL, mnist_path)
    with np.load(mnist_path) as f:
        return f['x_train'], f['y_train'], f['x_test'], f['y_test']


def load_mnist_binary(
    digit1: int = 3,
    digit2: int = 6,
    train_size: int = 1000,
    test_size: int = 200,
    data_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST and filter to a binary pair of digits.

    Args:
        digit1: First digit class (default: 3)
        digit2: Second digit class (default: 6)
        train_size: Number of training samples (default: 1000)
        test_size: Number of test samples (default: 200)
        data_seed: Seed for data subsampling only (default: None = nondeterministic)

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
        - X: full-resolution 28x28 images flattened to 784, normalized to [0, 1]
        - y: binary labels in {0, 1}
    """
    rng = np.random.default_rng(data_seed)

    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data' / 'raw'
    data_dir.mkdir(parents=True, exist_ok=True)

    mnist_path = data_dir / 'mnist.npz'
    x_train, y_train, x_test, y_test = _load_mnist_raw(mnist_path)

    train_filter = np.isin(y_train, [digit1, digit2])
    test_filter = np.isin(y_test, [digit1, digit2])

    x_train = x_train[train_filter]
    y_train = y_train[train_filter]
    x_test = x_test[test_filter]
    y_test = y_test[test_filter]

    y_train = (y_train == digit2).astype(int)
    y_test = (y_test == digit2).astype(int)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)

    if len(x_train) > train_size:
        train_indices = rng.choice(len(x_train), train_size, replace=False)
        x_train = x_train[train_indices]
        y_train = y_train[train_indices]

    if len(x_test) > test_size:
        test_indices = rng.choice(len(x_test), test_size, replace=False)
        x_test = x_test[test_indices]
        y_test = y_test[test_indices]

    return x_train, y_train, x_test, y_test


def _downsample_flatten(
    images: np.ndarray, image_size: Tuple[int, int]
) -> np.ndarray:
    """Downsample flattened 28x28 images to ``image_size`` (bilinear) and flatten."""
    images = images.reshape(len(images), 28, 28)
    h, w = image_size
    zoom_factor = (h / 28.0, w / 28.0)
    resized = np.stack([zoom(img, zoom_factor, order=1) for img in images])
    resized = resized.reshape(len(images), h * w)
    return resized


def prepare_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int,
    image_size: Tuple[int, int] = (4, 4),
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Reduce full-resolution MNIST features to ``n_components`` PCA features.

    Pipeline: 28x28 -> 4x4 bilinear downsample -> flatten (16 dims) -> PCA
    fitted on the training split only -> min-max normalized using training
    statistics only. Nothing from the test split influences the transform.

    Args:
        X_train: Training images, shape (n_train, 784), values in [0, 1]
        X_test: Test images, shape (n_test, 784), values in [0, 1]
        n_components: Number of PCA components (must equal number of qubits)
        image_size: Target downsample size (default: (4, 4))

    Returns:
        (X_train_r, X_test_r, pca_info)
        - X_train_r, X_test_r: PCA-reduced and normalized, shape (n, n_components)
        - pca_info: dict of fit statistics for the record

    Raises:
        ValueError: If inputs are not 784-dimensional or n_components is invalid.
    """
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    if n_components > DOWNSAMPLED_FEATURES:
        raise ValueError(
            f"n_components ({n_components}) cannot exceed the number of "
            f"downsampled features ({DOWNSAMPLED_FEATURES})"
        )
    if X_train.shape[1] != MNIST_FLAT_DIM or X_test.shape[1] != MNIST_FLAT_DIM:
        raise ValueError(
            f"Expected {MNIST_FLAT_DIM}-dim (28x28) input, got train={X_train.shape[1]} "
            f"test={X_test.shape[1]}"
        )

    train_16 = _downsample_flatten(X_train, image_size)
    test_16 = _downsample_flatten(X_test, image_size)

    pca = PCA(n_components=n_components)
    pca.fit(train_16)

    X_train_pc = pca.transform(train_16)
    X_test_pc = pca.transform(test_16)

    train_min = X_train_pc.min(axis=0)
    train_span = X_train_pc.max(axis=0) - train_min
    train_span = np.where(train_span == 0, 1.0, train_span)

    X_train_norm = (X_train_pc - train_min) / train_span
    X_test_norm = (X_test_pc - train_min) / train_span

    pca_info = {
        'n_components': int(n_components),
        'image_size': list(image_size),
        'explained_variance_ratio': [float(v) for v in pca.explained_variance_ratio_],
        'cumulative_explained_variance': float(pca.explained_variance_ratio_.sum()),
        'train_min': train_min.tolist(),
        'train_span': train_span.tolist(),
    }

    return X_train_norm, X_test_norm, pca_info


def encode_data_for_qnn(data: np.ndarray) -> np.ndarray:
    """
    Scale normalized features to rotation angles in [0, pi].

    Args:
        data: Normalized features, shape (n_samples, n_features), values in [0, 1]

    Returns:
        Encoded angles in [0, pi]
    """
    return data * np.pi


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist_binary(data_seed=42)
    X_train, X_test, info = prepare_features(X_train, X_test, n_components=4)
    print(f"Training set: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Test set: {X_test.shape}, Labels: {y_test.shape}")
    print(f"PCA info: n_components={info['n_components']}, "
          f"cumulative variance={info['cumulative_explained_variance']:.4f}")
