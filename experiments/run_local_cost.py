"""
Local Cost Experiment: Training with Per-Qubit Measurements

Runs a single local cost function experiment adapting the theoretical framework
of Cerezo et al. (2021). Uses averaged per-qubit Pauli-Z measurements instead
of a global cost function to maintain polynomial gradient scaling.

Usage:
    python experiments/run_local_cost.py configs/local_cost_4layer.yaml --seed 42
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import json
import argparse
import tensorflow_quantum as tfq
import cirq
import numpy as np
from pathlib import Path

from src.data import load_mnist_binary
from src.training import BaselineTrainer
from src.evaluation import plot_training_history


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Flatten nested config structure for easier access
    flat_config = {
        'experiment_name': config['experiment']['name'],
        'approach': config['experiment']['approach'],
        'n_qubits': config['model']['n_qubits'],
        'n_layers': config['model']['n_layers'],
        'learning_rate': config['training']['learning_rate'],
        'batch_size': config['training']['batch_size'],
        'epochs': config['training']['epochs'],
        'local_cost': config['training']['local_cost'],
        'digit1': config['data']['digit1'],
        'digit2': config['data']['digit2'],
        'train_size': config['data']['train_size'],
        'test_size': config['data']['test_size'],
        'image_size': config['data']['image_size'],
        'results_dir': config['output']['results_dir'],
        'random_seeds': config['random_seeds']
    }
    return flat_config, config  # Return both flat and original


def convert_to_circuits(data: np.ndarray, n_qubits: int = 4):
    """Convert data to quantum circuits."""
    qubits = cirq.GridQubit.rect(1, n_qubits)
    circuits = []
    
    for sample in data:
        circuit = cirq.Circuit()
        angles = sample * np.pi
        for i, qubit in enumerate(qubits):
            if i < len(angles):
                circuit.append(cirq.ry(angles[i])(qubit))
        circuits.append(circuit)
    
    return tfq.convert_to_tensor(circuits)


def main():
    """Run local cost experiment."""
    parser = argparse.ArgumentParser(description='Run local cost QNN experiment')
    parser.add_argument('config', type=str, help='Path to config YAML file')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides config)')
    args = parser.parse_args()
    
    print("="*70)
    print("LOCAL COST EXPERIMENT: Training with Per-Qubit Measurements")
    print("="*70)
    
    # Load configuration
    config, full_config = load_config(args.config)
    
    # Use provided seed or first seed from config
    seed = args.seed if args.seed is not None else config['random_seeds'][0]
    
    print(f"\nConfiguration: {args.config}")
    print(f"  Layers: {config['n_layers']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Local Cost: {config['local_cost']}")
    print(f"  Random Seed: {seed}")
    
    # Load and prepare data
    print("\nLoading MNIST data...")
    X_train, y_train, X_test, y_test = load_mnist_binary(
        digit1=config['digit1'],
        digit2=config['digit2'],
        train_size=config['train_size'],
        test_size=config['test_size'],
        image_size=tuple(config['image_size']),
        seed=seed
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Convert to quantum circuits
    print("\nConverting data to quantum circuits...")
    train_circuits = convert_to_circuits(X_train, n_qubits=config['n_qubits'])
    test_circuits = convert_to_circuits(X_test, n_qubits=config['n_qubits'])
    
    # Initialize trainer with local cost function
    print("\nInitializing trainer with local cost functions...")
    trainer = BaselineTrainer(
        n_qubits=config['n_qubits'],
        n_layers=config['n_layers'],
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        local_cost=True,  # KEY DIFFERENCE
        seed=seed
    )
    
    # Train model
    print("\nStarting training with local cost functions...")
    results = trainer.train(
        train_circuits=train_circuits,
        train_labels=y_train,
        val_circuits=test_circuits,
        val_labels=y_test,
        epochs=config['epochs']
    )
    
    # Create results directory with depth/seed structure
    results_dir = Path(config['results_dir']) / f"seed_{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_path = results_dir / "metrics.json"
    save_results = {
        'config': full_config,  # Save full config
        'seed': int(seed),
        'final_train_loss': float(results['final_train_loss']),
        'final_train_acc': float(results['final_train_acc']),
        'final_val_loss': float(results['final_val_loss']),
        'final_val_acc': float(results['final_val_acc']),
        'test_loss': float(results['test_loss']),
        'test_acc': float(results['test_acc']),
        'training_time': float(results['training_time']),
        'gradient_stats': {k: float(v) for k, v in results['gradient_stats'].items()},
        'barren_plateau_detected': bool(results['barren_plateau_detected']),
        'history': {k: [float(v) for v in vals] for k, vals in results['history'].items()}
    }
    with open(metrics_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Plot and save training history
    plot_path = results_dir / "training_history.png"
    plot_training_history(
        results['history'],
        save_path=str(plot_path),
        title=f"Local Cost Training - {config['n_layers']} Layers (Seed {seed})"
    )
    print(f"Plot saved to {plot_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Test Accuracy: {results['test_acc']*100:.2f}%")
    print(f"Training Time: {results['training_time']:.2f}s")
    print(f"Barren Plateau Detected: {results['barren_plateau_detected']}")
    print(f"Mean Gradient Norm: {results['gradient_stats']['mean_norm']:.6e}")
    print("="*70)


if __name__ == "__main__":
    main()
