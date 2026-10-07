"""Shared helpers for the experiment runners.

Each runner (``run_baseline.py``, ``run_layerwise.py``, ``run_local_cost.py``)
loads a config, derives the (data, init, training) seed triple for one seed
index, loads/encodes data through the fixed PCA pipeline, trains for exactly
``total_updates`` gradient steps, and persists the metrics schema plus the
canonical ``training_history.png``.
"""

import os

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1') 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cirq
import tensorflow_quantum as tfq

from src.data import load_mnist_binary, prepare_features, encode_data_for_qnn
from src.evaluation import plot_training_history


def convert_to_circuits(data: np.ndarray, n_qubits: int):
    """Encode exactly ``n_qubits`` features with an ``ry(pi*x)`` gate each.

    Fails loudly if the feature count does not match the qubit count
    (regression guard against silent truncation).
    """
    assert data.shape[1] == n_qubits, (
        f"Feature count {data.shape[1]} must equal qubit count {n_qubits}"
    )
    qubits = cirq.GridQubit.rect(1, n_qubits)
    circuits = []
    for sample in data:
        circuit = cirq.Circuit()
        angles = encode_data_for_qnn(sample)
        for qubit, angle in zip(qubits, angles):
            circuit.append(cirq.ry(angle)(qubit))
        circuits.append(circuit)
    return tfq.convert_to_tensor(circuits)


def load_and_prepare(config: dict, data_seed: int):
    """Load MNIST through the fixed PCA pipeline and encode into circuits.

    Returns:
        (train_circuits, y_train, test_circuits, y_test, pca_info)
    """
    X_train, y_train, X_test, y_test = load_mnist_binary(
        digit1=config['data']['digit1'],
        digit2=config['data']['digit2'],
        train_size=config['data']['train_size'],
        test_size=config['data']['test_size'],
        data_seed=data_seed,
    )
    X_train, X_test, pca_info = prepare_features(
        X_train, X_test,
        n_components=config['data']['n_components'],
        image_size=tuple(config['data']['image_size']),
    )
    train_circuits = convert_to_circuits(X_train, n_qubits=config['model']['n_qubits'])
    test_circuits = convert_to_circuits(X_test, n_qubits=config['model']['n_qubits'])
    return train_circuits, y_train, test_circuits, y_test, pca_info


def _jsonify(obj):
    """Recursively convert numpy types to JSON-native Python types."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_run(
    results: dict,
    results_dir: str,
    seed_index: int,
    seed_triple: dict,
    pca_info: dict,
):
    """Persist a metrics.json and the canonical training_history.png.

    Args:
        results: Trainer results dict (incl. ``config`` and ``history``).
        results_dir: Config ``output.results_dir``.
        seed_index: Seed-triple index.
        seed_triple: Derived (data_seed, init_seed, training_seed).
        pca_info: PCA fit statistics from ``prepare_features``.

    Returns:
        (metrics_path, plot_path)
    """
    out_dir = Path(results_dir) / f'seed_{seed_index}'
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / 'metrics.json'
    payload = {
        **results,
        'data_seed': int(seed_triple['data_seed']),
        'init_seed': int(seed_triple['init_seed']),
        'training_seed': int(seed_triple['training_seed']),
        'seed_index': int(seed_index),
        'pca_info': _jsonify(pca_info),
    }
    with open(metrics_path, 'w') as f:
        json.dump(payload, f, indent=2, default=_jsonify)

    plot_path = out_dir / 'training_history.png'
    plot_training_history(
        results['history'],
        save_path=str(plot_path),
        title=(
            f"{results['config']['approach']} training - "
            f"{results['config']['n_layers']} layers (seed index {seed_index})"
        ),
        show=False,
    )
    return metrics_path, plot_path


def print_run_header(approach: str, config: dict, seed_index: int, seed_triple: dict) -> None:
    """Print a concise summary of the run configuration."""
    training = config['training']
    model = config['model']
    data = config['data']
    print("=" * 70)
    print(f"{approach.upper()} EXPERIMENT")
    print("=" * 70)
    print(f"  Layers: {model['n_layers']}   Qubits: {model['n_qubits']}")
    print(f"  Cost function: {training['cost_function']}")
    print(f"  Total updates: {training['total_updates']}")
    print(f"  Learning rate: {training['learning_rate']}   Batch size: {training['batch_size']}")
    print(f"  Seed index: {seed_index}")
    print(f"  data_seed={seed_triple['data_seed']}  init_seed={seed_triple['init_seed']}  "
          f"training_seed={seed_triple['training_seed']}")
    print(f"  Train size: {data['train_size']}   Test size: {data['test_size']}")


def run_is_complete(results_dir: str, seed_index: int) -> bool:
    """True if a finished metrics.json already exists for this seed index."""
    return (Path(results_dir) / f'seed_{seed_index}' / 'metrics.json').exists()


def seed_checkpoint_dir(results_dir: str, seed_index: int) -> Path:
    """Per-seed checkpoint directory (created by the trainer on demand)."""
    return Path(results_dir) / f'seed_{seed_index}' / 'checkpoint'