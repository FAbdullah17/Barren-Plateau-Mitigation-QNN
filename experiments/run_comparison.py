"""
Comprehensive Comparison: All Three Approaches
Run baseline, layerwise, and local cost experiments and compare results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import json
import tensorflow_quantum as tfq
import cirq
import numpy as np
from pathlib import Path
import pandas as pd

from src.data import load_mnist_binary
from src.training import BaselineTrainer, LayerwiseTrainer
from src.evaluation import plot_comparison, plot_gradient_trajectory


def load_config(config_path: str = "configs/multi_depth.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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


def run_single_experiment(approach: str, config: dict, train_circuits, train_labels, 
                          test_circuits, test_labels, n_layers: int, seed: int):
    """Run a single experiment with given configuration."""
    print(f"\nRunning {approach} with {n_layers} layers, seed {seed}...")
    
    if approach == "baseline":
        trainer = BaselineTrainer(
            n_qubits=config['n_qubits'],
            n_layers=n_layers,
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            local_cost=False,
            seed=seed
        )
        results = trainer.train(
            train_circuits=train_circuits,
            train_labels=train_labels,
            val_circuits=test_circuits,
            val_labels=test_labels,
            epochs=config['epochs']
        )
    
    elif approach == "layerwise":
        trainer = LayerwiseTrainer(
            n_qubits=config['n_qubits'],
            target_layers=n_layers,
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            epochs_per_layer=config['epochs_per_layer'],
            finetune_epochs=config['finetune_epochs'],
            local_cost=False,
            seed=seed
        )
        results = trainer.train(
            train_circuits=train_circuits,
            train_labels=train_labels,
            val_circuits=test_circuits,
            val_labels=test_labels
        )
    
    elif approach == "local_cost":
        trainer = BaselineTrainer(
            n_qubits=config['n_qubits'],
            n_layers=n_layers,
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            local_cost=True,
            seed=seed
        )
        results = trainer.train(
            train_circuits=train_circuits,
            train_labels=train_labels,
            val_circuits=test_circuits,
            val_labels=test_labels,
            epochs=config['epochs']
        )
    
    return results


def main():
    """Run comprehensive comparison."""
    print("="*70)
    print("COMPREHENSIVE COMPARISON: All Approaches")
    print("="*70)
    
    # Load configuration
    config = load_config()
    print("\nConfiguration:")
    print(f"  Depths: {config['depths']}")
    print(f"  Seeds: {config['seeds']}")
    print(f"  Qubits: {config['n_qubits']}")
    
    # Create results directory
    results_dir = Path("results/comparison")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Store all results
    all_results = []
    
    # Run experiments for each depth
    for n_layers in config['depths']:
        print(f"\n{'='*70}")
        print(f"TESTING DEPTH: {n_layers} LAYERS")
        print(f"{'='*70}")
        
        for seed in config['seeds']:
            print(f"\nSeed: {seed}")
            
            # Load data with this seed
            X_train, y_train, X_test, y_test = load_mnist_binary(
                digit1=config['digit1'],
                digit2=config['digit2'],
                train_size=config['train_size'],
                test_size=config['test_size'],
                image_size=tuple(config['image_size']),
                seed=seed
            )
            
            # Convert to circuits
            train_circuits = convert_to_circuits(X_train, n_qubits=config['n_qubits'])
            test_circuits = convert_to_circuits(X_test, n_qubits=config['n_qubits'])
            
            # Run all three approaches
            for approach in ["baseline", "layerwise", "local_cost"]:
                try:
                    results = run_single_experiment(
                        approach=approach,
                        config=config,
                        train_circuits=train_circuits,
                        train_labels=y_train,
                        test_circuits=test_circuits,
                        test_labels=y_test,
                        n_layers=n_layers,
                        seed=seed
                    )
                    
                    # Store result
                    all_results.append({
                        'approach': approach,
                        'n_layers': n_layers,
                        'seed': seed,
                        'test_acc': results['test_acc'] * 100,
                        'training_time': results['training_time'],
                        'mean_grad_norm': results['gradient_stats']['mean_norm'],
                        'barren_plateau': results['barren_plateau_detected']
                    })
                    
                except Exception as e:
                    print(f"ERROR in {approach}: {e}")
                    all_results.append({
                        'approach': approach,
                        'n_layers': n_layers,
                        'seed': seed,
                        'test_acc': 0.0,
                        'training_time': 0.0,
                        'mean_grad_norm': 0.0,
                        'barren_plateau': True,
                        'error': str(e)
                    })
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Save raw results
    df.to_csv(results_dir / "all_results.csv", index=False)
    print(f"\nRaw results saved to {results_dir / 'all_results.csv'}")
    
    # Compute summary statistics
    summary = df.groupby(['approach', 'n_layers']).agg({
        'test_acc': ['mean', 'std', 'max', 'min'],
        'training_time': ['mean', 'std'],
        'mean_grad_norm': ['mean', 'std'],
        'barren_plateau': 'sum'
    }).round(4)
    
    # Save summary
    summary.to_csv(results_dir / "summary_statistics.csv")
    print(f"Summary statistics saved to {results_dir / 'summary_statistics.csv'}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(summary)
    
    # Compute success rates
    print("\n" + "="*70)
    print("SUCCESS RATES (Test Accuracy >= 90%)")
    print("="*70)
    success_threshold = config.get('success_threshold', 90.0)
    success_rates = df.groupby(['approach', 'n_layers']).apply(
        lambda x: (x['test_acc'] >= success_threshold).sum() / len(x) * 100
    )
    print(success_rates)
    
    # Save final report
    with open(results_dir / "final_report.txt", 'w') as f:
        f.write("="*70 + "\n")
        f.write("COMPREHENSIVE COMPARISON REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total experiments run: {len(all_results)}\n")
        f.write(f"Depths tested: {config['depths']}\n")
        f.write(f"Seeds used: {config['seeds']}\n\n")
        f.write("Summary Statistics:\n")
        f.write(summary.to_string())
        f.write("\n\nSuccess Rates:\n")
        f.write(success_rates.to_string())
    
    print(f"\nFinal report saved to {results_dir / 'final_report.txt'}")
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
