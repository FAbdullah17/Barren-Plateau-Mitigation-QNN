"""
Layerwise Experiment: Incremental Layer-by-Layer Training
Implements Skolik et al. (2020) methodology.

Developer Assignment:
    Implementation (Weeks 3-4): Asma Zubair
    Testing (Weeks 5-6): All team members
    Experiments (Weeks 7-9):
        - Asma Zubair: 4-layer layerwise (Week 7, 5 seeds)
        - Frahan Riaz: 6-layer layerwise (Week 8, 5 seeds)
        - Fahad Abdullah: 8-layer layerwise (Week 9, 5 seeds) [KEY FINDING]
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
from src.training import LayerwiseTrainer
from src.evaluation import plot_training_history


def load_config(config_path: str = "configs/layerwise_test.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Flatten nested config structure for easier access
    flat_config = {
        'experiment_name': config['experiment']['name'],
        'approach': config['experiment']['approach'],
        'n_qubits': config['model']['n_qubits'],
        'target_layers': config['model']['target_layers'],
        'learning_rate': config['training']['learning_rate'],
        'batch_size': config['training']['batch_size'],
        'epochs_per_layer': config['training']['epochs_per_layer'],
        'finetune_epochs': config['training']['finetune_epochs'],
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
    """Run layerwise experiment."""
    parser = argparse.ArgumentParser(description='Run layerwise QNN experiment')
    parser.add_argument('config', type=str, help='Path to config YAML file')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides config)')
    args = parser.parse_args()
    
    print("="*70)
    print("LAYERWISE EXPERIMENT: Incremental Layer-by-Layer Training")
    print("="*70)
    
    # Load configuration
    config, full_config = load_config(args.config)
    
    # Use provided seed or first seed from config
    seed = args.seed if args.seed is not None else config['random_seeds'][0]
    
    print(f"\nConfiguration: {args.config}")
    print(f"  Target Layers: {config['target_layers']}")
    print(f"  Epochs per Layer: {config['epochs_per_layer']}")
    print(f"  Finetune Epochs: {config['finetune_epochs']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Batch Size: {config['batch_size']}")
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
    
    # Initialize trainer
    print("\nInitializing layerwise trainer...")
    trainer = LayerwiseTrainer(
        n_qubits=config['n_qubits'],
        target_layers=config['target_layers'],
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        epochs_per_layer=config['epochs_per_layer'],
        finetune_epochs=config['finetune_epochs'],
        local_cost=config['local_cost'],
        seed=seed
    )
    
    # Train model
    print("\nStarting layerwise training...")
    results = trainer.train(
        train_circuits=train_circuits,
        train_labels=y_train,
        val_circuits=test_circuits,
        val_labels=y_test
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
        'gradient_stats': results['gradient_stats'],
        'barren_plateau_detected': bool(results['barren_plateau_detected'])
    }
    with open(metrics_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Plot and save training history
    plot_path = results_dir / "training_history.png"
    plot_training_history(
        results['history'],
        save_path=str(plot_path),
        title=f"Layerwise Training - {config['target_layers']} Layers (Seed {seed})"
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
