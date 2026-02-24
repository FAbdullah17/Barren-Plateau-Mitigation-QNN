"""Visualization utilities for training results and cross-approach comparison.

Provides plotting functions for training history (loss, accuracy, gradient
norms, gradient variance), multi-approach comparison bar charts, and
gradient norm trajectory overlays. All plots use publication-quality
formatting with Seaborn styling.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
import os


# Set publication-quality style defaults
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None,
    title: str = "Training History"
):
    """
    Plot training and validation metrics over epochs.
    
    Generates a 2x2 grid showing loss, accuracy, gradient norms (log scale),
    and gradient variance (log scale). Optionally marks layer transitions
    for layerwise training experiments.
    
    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 'train_acc',
                 'val_acc', 'gradient_norms', 'gradient_variance', and
                 optionally 'layer_transitions'.
        save_path: Path to save figure (creates directories if needed).
        title: Figure title.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gradient norms (log scale)
    axes[1, 0].plot(history['gradient_norms'], linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('Gradient Norms (Log Scale)')
    axes[1, 0].axhline(y=1e-6, color='red', linestyle='--', label='Barren Plateau Threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gradient variance (log scale)
    axes[1, 1].plot(history['gradient_variance'], linewidth=2, color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Gradient Variance')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('Gradient Variance (Log Scale)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Mark layer transitions for layerwise training
    if 'layer_transitions' in history and history['layer_transitions']:
        for transition in history['layer_transitions']:
            for ax in axes.flat:
                ax.axvline(x=transition, color='orange', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_comparison(
    results_dict: Dict[str, Dict],
    save_path: Optional[str] = None
):
    """
    Compare metrics across different training approaches.
    
    Generates a 1x3 figure showing test accuracy, training time, and
    mean gradient norm for each approach side by side.
    
    Args:
        results_dict: Dictionary mapping approach names to result dictionaries,
                      each containing 'test_acc', 'training_time', and
                      'gradient_stats'.
        save_path: Path to save figure.
    """
    approaches = list(results_dict.keys())
    test_accs = [results_dict[a]['test_acc'] * 100 for a in approaches]
    train_times = [results_dict[a]['training_time'] for a in approaches]
    grad_norms = [results_dict[a]['gradient_stats'].get('mean_norm', 0) for a in approaches]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Comparison Across Approaches', fontsize=16, fontweight='bold')
    
    # Test Accuracy
    bars1 = axes[0].bar(approaches, test_accs, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Test Accuracy')
    axes[0].set_ylim([0, 100])
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
    
    # Training Time
    bars2 = axes[1].bar(approaches, train_times, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[1].set_ylabel('Training Time (seconds)')
    axes[1].set_title('Training Time')
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}s', ha='center', va='bottom')
    
    # Gradient Norms (log scale)
    bars3 = axes[2].bar(approaches, grad_norms, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[2].set_ylabel('Mean Gradient Norm')
    axes[2].set_title('Gradient Behavior')
    axes[2].set_yscale('log')
    axes[2].axhline(y=1e-6, color='red', linestyle='--', linewidth=2, label='BP Threshold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    plt.show()


def plot_gradient_trajectory(
    histories: Dict[str, Dict],
    save_path: Optional[str] = None
):
    """
    Plot gradient norm trajectories for different training approaches.
    
    Overlays gradient norm curves on a single log-scale plot to highlight
    differences in gradient behavior between approaches (e.g., vanishing
    gradients in baseline vs. maintained gradients in layerwise training).
    
    Args:
        histories: Dictionary mapping approach names to training histories,
                   each containing a 'gradient_norms' list.
        save_path: Path to save figure.
    """
    plt.figure(figsize=(12, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, (name, history) in enumerate(histories.items()):
        plt.plot(
            history['gradient_norms'],
            label=name,
            linewidth=2,
            color=colors[i % len(colors)]
        )
    
    plt.axhline(y=1e-6, color='red', linestyle='--', linewidth=2, 
                label='Barren Plateau Threshold', alpha=0.7)
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Gradient Norm', fontsize=12)
    plt.title('Gradient Norm Trajectories', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved gradient trajectory plot to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Verify plotting with synthetic data
    history = {
        'train_loss': np.random.rand(50) * 2,
        'val_loss': np.random.rand(50) * 2,
        'train_acc': np.random.rand(50) * 100,
        'val_acc': np.random.rand(50) * 100,
        'gradient_norms': np.exp(-np.linspace(0, 5, 50)) * 0.1,
        'gradient_variance': np.exp(-np.linspace(0, 5, 50)) * 0.01
    }
    
    plot_training_history(history, title="Test Plot")
