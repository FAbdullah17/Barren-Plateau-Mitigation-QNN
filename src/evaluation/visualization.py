"""Visualization utilities for training results and cross-approach comparison.

Plots step-based training history, the ``mean_param_grad_variance`` trajectory
per approach, and a cross-approach test-accuracy comparison across depths. No
binary "barren plateau" threshold lines are drawn.
"""

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.utils.constants import COLORS, PLOT_DPI, PLOT_FIGSIZE

# Publication-quality defaults.
plt.rcParams['figure.figsize'] = PLOT_FIGSIZE
plt.rcParams['font.size'] = 10

_FALLBACK_COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None,
    title: str = "Training History",
    show: bool = True,
):
    """Plot step-based training loss and accuracy.

    The history contains per-step ``train_loss``/``train_acc`` (x = ``step``)
    and validation metrics recorded every ``log_frequency`` steps
    (x = ``val_step``).

    Args:
        history: dict with ``step``, ``train_loss``, ``train_acc``,
            ``val_step``, ``val_loss``, ``val_acc``.
        save_path: Path to save the figure (directories created as needed).
        title: Figure title.
        show: Call ``plt.show()`` when True (pass False in batch runners).
    """
    step = np.asarray(history['step'], dtype=int)
    val_step = np.asarray(history.get('val_step', []), dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    axes[0].plot(step, history['train_loss'], label='Train', linewidth=2)
    if len(val_step):
        axes[0].plot(
            val_step, history['val_loss'], label='Validation',
            linewidth=2, marker='o', markersize=3,
        )
    axes[0].set_xlabel('Gradient step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss over gradient steps')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(step, history['train_acc'], label='Train', linewidth=2)
    if len(val_step):
        axes[1].plot(
            val_step, history['val_acc'], label='Validation',
            linewidth=2, marker='o', markersize=3,
        )
    axes[1].set_xlabel('Gradient step')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy over gradient steps')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_gradient_trajectory(
    trajectories: Dict[str, Dict],
    save_path: Optional[str] = None,
    title: str = "Mean parameter-gradient variance trajectories",
    show: bool = True,
):
    """Overlay per-approach ``mean_param_grad_variance`` trajectories.

    The training diagnostic ``\\bar{V}^x`` (variance over samples of the
    per-parameter gradient) is plotted against the gradient step on a log
    scale, one curve per approach.

    Args:
        trajectories: dict mapping approach name to a ``training_diagnostic``
            dict containing ``trajectory: {step: [...], mean_param_grad_variance: [...]}``.
        save_path: Path to save the figure.
        title: Figure title.
        show: Call ``plt.show()`` when True.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (name, diagnostic) in enumerate(trajectories.items()):
        trajectory = diagnostic['trajectory']
        color = COLORS.get(name, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])
        ax.plot(
            trajectory['step'],
            trajectory['mean_param_grad_variance'],
            label=name, linewidth=2, color=color,
        )

    ax.set_xlabel('Gradient step')
    ax.set_ylabel(r'Mean parameter-gradient variance $\bar{V}^x$')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"Saved gradient trajectory plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_comparison(
    summary: Dict[str, Dict],
    save_path: Optional[str] = None,
    title: str = "Mean test accuracy across depths",
    show: bool = True,
):
    """Grouped bar chart of mean test accuracy per approach per depth.

    Args:
        summary: dict mapping approach name to ``{str(depth): {'test_acc': mean,
            'test_acc_sd': sd, 'test_acc_ci': [lo, hi]}}`` (produced by
            ``run_comparison.py``).
        save_path: Path to save the figure.
        title: Figure title.
        show: Call ``plt.show()`` when True.
    """
    approaches = list(summary.keys())
    depths = sorted({int(d) for entry in summary.values() for d in entry})

    x = np.arange(len(depths))
    width = 0.8 / len(approaches)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, approach in enumerate(approaches):
        color = COLORS.get(approach, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])
        means = []
        yerrs = []
        for d in depths:
            entry = summary[approach][str(d)]
            means.append(entry['test_acc'])
            ci = entry['test_acc_ci']
            yerrs.append([entry['test_acc'] - ci[0], ci[1] - entry['test_acc']])
        ax.bar(
            x + i * width, means, width, label=approach,
            color=color, yerr=np.asarray(yerrs).T, capsize=3,
        )

    ax.set_xticks(x + width * (len(approaches) - 1) / 2)
    ax.set_xticklabels([f'L={d}' for d in depths])
    ax.set_ylabel('Mean test accuracy')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


if __name__ == "__main__":
    # Verify plotting with synthetic data following the metrics schema.
    rng = np.random.default_rng(0)
    n_steps = 50
    history = {
        'step': list(range(n_steps)),
        'train_loss': list(0.7 * np.exp(-0.05 * np.arange(n_steps)) + 0.1 * rng.random(n_steps)),
        'train_acc': list(0.5 + 0.4 * (1 - np.exp(-0.05 * np.arange(n_steps)))),
        'val_step': list(range(0, n_steps, 10)),
        'val_loss': [0.7, 0.6, 0.5, 0.45, 0.4],
        'val_acc': [0.55, 0.65, 0.72, 0.78, 0.82],
    }
    plot_training_history(history, title="Synthetic training history")