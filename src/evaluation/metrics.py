"""Metrics and evaluation utilities.

Developer Assignment (Weeks 1-2):
    Primary: Asma Zubair - Gradient tracking & validation metrics
    Support: Fahad Abdullah - Gradient analysis tools
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class GradientStatistics:
    """Container for gradient statistics."""
    mean_norm: float
    std_norm: float
    variance: float
    min_norm: float
    max_norm: float
    median_norm: float


class GradientTracker:
    """Track and analyze gradients during training."""
    
    def __init__(self, barren_plateau_threshold: float = 1e-6):
        """
        Initialize gradient tracker.
        
        Args:
            barren_plateau_threshold: Threshold for detecting barren plateaus
        """
        self.gradient_norms = []
        self.barren_plateau_threshold = barren_plateau_threshold
    
    def update(self, gradients: List[np.ndarray]):
        """
        Update with new gradients.
        
        Args:
            gradients: List of gradient arrays
        """
        norms = [np.linalg.norm(g) for g in gradients]
        self.gradient_norms.extend(norms)
    
    def get_statistics(self) -> Dict:
        """Compute gradient statistics."""
        if not self.gradient_norms:
            return {}
        
        norms = np.array(self.gradient_norms)
        
        return {
            'mean_norm': float(np.mean(norms)),
            'std_norm': float(np.std(norms)),
            'variance': float(np.var(norms)),
            'min_norm': float(np.min(norms)),
            'max_norm': float(np.max(norms)),
            'median_norm': float(np.median(norms)),
            'total_updates': len(norms)
        }
    
    def detect_barren_plateau(self, window_size: int = 10) -> bool:
        """
        Detect if barren plateau is present.
        
        Args:
            window_size: Number of recent gradients to check
            
        Returns:
            True if barren plateau detected
        """
        if len(self.gradient_norms) < window_size:
            return False
        
        recent_norms = self.gradient_norms[-window_size:]
        mean_recent_norm = np.mean(recent_norms)
        
        return mean_recent_norm < self.barren_plateau_threshold
    
    def get_variance_trajectory(self, window_size: int = 50) -> List[float]:
        """
        Compute rolling variance of gradient norms.
        
        Args:
            window_size: Window size for rolling variance
            
        Returns:
            List of variance values
        """
        if len(self.gradient_norms) < window_size:
            return []
        
        variances = []
        for i in range(window_size, len(self.gradient_norms) + 1):
            window = self.gradient_norms[i-window_size:i]
            variances.append(np.var(window))
        
        return variances


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Args:
        predictions: Model predictions (probabilities)
        labels: Ground truth labels
        
    Returns:
        Accuracy as percentage
    """
    pred_labels = (predictions > 0.5).astype(int)
    correct = np.sum(pred_labels == labels)
    return (correct / len(labels)) * 100.0


def compute_success_rate(accuracies: List[float], threshold: float = 90.0) -> float:
    """
    Compute success rate across multiple runs.
    
    Args:
        accuracies: List of test accuracies from different runs
        threshold: Minimum accuracy to count as success
        
    Returns:
        Success rate as percentage
    """
    successes = sum(1 for acc in accuracies if acc >= threshold)
    return (successes / len(accuracies)) * 100.0


def compare_approaches(results_dict: Dict[str, Dict]) -> Dict:
    """
    Compare results from different training approaches.
    
    Args:
        results_dict: Dictionary mapping approach names to result dictionaries
        
    Returns:
        Comparison summary
    """
    comparison = {}
    
    for name, results in results_dict.items():
        comparison[name] = {
            'test_accuracy': results['test_acc'],
            'training_time': results['training_time'],
            'final_gradient_norm': results['gradient_stats'].get('mean_norm', 0),
            'barren_plateau': results['barren_plateau_detected']
        }
    
    # Find best approach
    best_acc = max(comparison.items(), key=lambda x: x[1]['test_accuracy'])
    fastest = min(comparison.items(), key=lambda x: x[1]['training_time'])
    
    comparison['summary'] = {
        'best_accuracy': best_acc[0],
        'fastest_training': fastest[0]
    }
    
    return comparison


if __name__ == "__main__":
    # Test gradient tracker
    tracker = GradientTracker()
    
    # Simulate gradients
    for i in range(100):
        fake_gradients = [np.random.randn(10) * (0.1 / (i + 1))]
        tracker.update(fake_gradients)
    
    stats = tracker.get_statistics()
    print("Gradient Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.6f}")
    
    print(f"\nBarren plateau detected: {tracker.detect_barren_plateau()}")
