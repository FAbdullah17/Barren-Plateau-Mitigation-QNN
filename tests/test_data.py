#!/usr/bin/env python
"""
Tests for the data pipeline: seed-decoupled loading, train-only PCA,
leakage-free normalization, and all-features encoding.

Run with: python -m pytest tests/test_data.py -v
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_mnist_binary, prepare_features, encode_data_for_qnn

N_QUBITS = 4


def test_loading_deterministic_with_same_data_seed():
    X1, y1, Xt1, yt1 = load_mnist_binary(train_size=100, test_size=50, data_seed=42)
    X2, y2, Xt2, yt2 = load_mnist_binary(train_size=100, test_size=50, data_seed=42)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)
    assert np.array_equal(Xt1, Xt2)
    assert np.array_equal(yt1, yt2)


def test_different_data_seeds_give_different_subsamples():
    X1, _, _, _ = load_mnist_binary(train_size=100, test_size=50, data_seed=42)
    X2, _, _, _ = load_mnist_binary(train_size=100, test_size=50, data_seed=123)
    assert not np.array_equal(X1, X2)


def test_loading_returns_full_resolution_normalized_data():
    X_train, y_train, X_test, y_test = load_mnist_binary(
        train_size=100, test_size=50, data_seed=42
    )
    assert X_train.shape == (100, 784)
    assert X_test.shape == (50, 784)
    assert y_train.shape == (100,)
    assert y_test.shape == (50,)
    assert X_train.min() >= 0.0 and X_train.max() <= 1.0
    assert X_test.min() >= 0.0 and X_test.max() <= 1.0
    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})


def test_loader_does_not_mutate_global_seed_state():
    state_before = np.random.get_state()
    load_mnist_binary(train_size=100, test_size=50, data_seed=42)
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    assert np.array_equal(state_before[1], state_after[1])
    assert state_before[2] == state_after[2]
    assert state_before[3] == state_after[3]
    assert state_before[4] == state_after[4]


def test_prepare_features_shapes_and_range():
    X_train, y_train, X_test, y_test = load_mnist_binary(
        train_size=100, test_size=50, data_seed=42
    )
    X_train_r, X_test_r, pca_info = prepare_features(
        X_train, X_test, n_components=N_QUBITS
    )
    assert X_train_r.shape == (100, N_QUBITS)
    assert X_test_r.shape == (50, N_QUBITS)
    # train values exactly map to [0, 1] by construction
    assert X_train_r.min() >= 0.0 and X_train_r.max() <= 1.0
    assert 'explained_variance_ratio' in pca_info
    assert 'cumulative_explained_variance' in pca_info
    assert pca_info['n_components'] == N_QUBITS
    assert len(pca_info['explained_variance_ratio']) == N_QUBITS
    assert len(pca_info['train_min']) == N_QUBITS
    assert len(pca_info['train_span']) == N_QUBITS


def test_pca_fit_on_train_only_no_leakage():
    X_train, _, X_test, _ = load_mnist_binary(train_size=100, test_size=50, data_seed=42)
    _, _, other_test, _ = load_mnist_binary(train_size=100, test_size=50, data_seed=789)

    X_train_r1, X_test_r1, info1 = prepare_features(
        X_train, X_test, n_components=N_QUBITS
    )
    X_train_r2, X_test_r2, info2 = prepare_features(
        X_train, other_test, n_components=N_QUBITS
    )

    # The transform depends only on the training split: swapping the test
    # split must not change the train transform.
    assert np.allclose(X_train_r1, X_train_r2)
    assert np.allclose(info1['train_min'], info2['train_min'])
    assert np.allclose(info1['train_span'], info2['train_span'])


def test_prepare_features_is_deterministic():
    X_train, _, X_test, _ = load_mnist_binary(train_size=100, test_size=50, data_seed=42)
    A1, B1, i1 = prepare_features(X_train, X_test, n_components=N_QUBITS)
    A2, B2, i2 = prepare_features(X_train, X_test, n_components=N_QUBITS)
    assert np.array_equal(A1, A2)
    assert np.array_equal(B1, B2)
    assert np.array_equal(i1['train_min'], i2['train_min'])


def test_prepare_features_rejects_bad_input():
    X_train, _, X_test, _ = load_mnist_binary(train_size=100, test_size=50, data_seed=42)
    try:
        prepare_features(X_train[:, :4], X_test, n_components=N_QUBITS)
        assert False, "expected ValueError for non-784 input"
    except ValueError:
        pass
    try:
        prepare_features(X_train, X_test, n_components=0)
        assert False, "expected ValueError for n_components < 1"
    except ValueError:
        pass
    try:
        prepare_features(X_train, X_test, n_components=32)
        assert False, "expected ValueError for n_components > downsampled features"
    except ValueError:
        pass


def test_encode_data_for_qnn_angles_in_range():
    data = np.array([[0.0, 0.25, 0.5, 1.0]])
    angles = encode_data_for_qnn(data)
    assert angles.shape == (1, 4)
    assert angles.min() >= 0.0 and angles.max() <= np.pi
    assert np.allclose(angles, data * np.pi)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
