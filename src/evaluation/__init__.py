"""Evaluation module initialization."""

from .metrics import (
    GradientTracker,
    compute_accuracy,
    compare_approaches,
    landscape_variance
)
from .visualization import (
    plot_training_history,
    plot_comparison,
    plot_gradient_trajectory
)
from .variance_scaling import (
    draw_parameter_vectors,
    fit_scaling,
    plot_variance_scaling,
    summarize_landscape
)

__all__ = [
    "GradientTracker",
    "compute_accuracy",
    "compare_approaches",
    "landscape_variance",
    "plot_training_history",
    "plot_comparison",
    "plot_gradient_trajectory",
    "draw_parameter_vectors",
    "fit_scaling",
    "plot_variance_scaling",
    "summarize_landscape"
]
