"""Baseline runner: standard end-to-end training with the global cost function.

Consumes exactly ``total_updates`` gradient steps with the
full ansatz optimized simultaneously. One run per seed-triple index.

Usage:
    python experiments/run_baseline.py configs/baseline_8layer.yaml --seed-index 0
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')  # TFQ requires Keras 2

import argparse

from experiments.common import (
    load_and_prepare, save_run, print_run_header,
)
from src.utils.config_validator import load_config, validate_config, derive_seed_triple
from src.training import BaselineTrainer


def main():
    parser = argparse.ArgumentParser(
        description='Run a baseline (global cost) QNN experiment'
    )
    parser.add_argument('config', type=str, help='Path to a config YAML')
    parser.add_argument(
        '--seed-index', type=int, default=0,
        help='Seed-triple index in [0, seeds.seed_triples)',
    )
    args = parser.parse_args()

    config = validate_config(load_config(args.config), approach='baseline')

    if not 0 <= args.seed_index < config['seeds']['seed_triples']:
        parser.error(
            f"--seed-index {args.seed_index} out of range "
            f"[0, {config['seeds']['seed_triples']})"
        )
    seed_triple = derive_seed_triple(config['seeds']['base_seed'], args.seed_index)

    print_run_header('baseline', config, args.seed_index, seed_triple)

    train_circuits, y_train, test_circuits, y_test, pca_info = load_and_prepare(
        config, seed_triple['data_seed']
    )

    trainer = BaselineTrainer(
        n_qubits=config['model']['n_qubits'],
        n_layers=config['model']['n_layers'],
        cost=config['training']['cost_function'],
        learning_rate=config['training']['learning_rate'],
        batch_size=config['training']['batch_size'],
        total_updates=config['training']['total_updates'],
        log_frequency=config['metrics']['log_frequency'],
        diagnostic_samples=config['metrics']['diagnostic_samples'],
        init_seed=seed_triple['init_seed'],
        training_seed=seed_triple['training_seed'],
        track_gradients=config['metrics']['track_gradients'],
    )

    results = trainer.train(
        train_circuits=train_circuits,
        train_labels=y_train,
        val_circuits=test_circuits,
        val_labels=y_test,
    )

    results['config']['approach'] = config['experiment']['approach']

    metrics_path, _ = save_run(
        results, config['output']['results_dir'], args.seed_index, seed_triple, pca_info
    )

    diagnostic = results['training_diagnostic']
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"  Test accuracy: {results['test_acc'] * 100:.2f}%")
    print(f"  Test loss:     {results['test_loss']:.4f}")
    print(f"  Training time: {results['training_time_seconds']:.2f}s")
    print(f"  Mean param-gradient variance: "
          f"{diagnostic['mean_param_grad_variance']:.3e}")
    print(f"  n_parameters:  {results['n_parameters']}")
    print(f"  Metrics saved to: {metrics_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()