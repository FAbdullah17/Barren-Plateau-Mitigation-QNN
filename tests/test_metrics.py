"""Unit tests for evaluation metrics."""

import pytest
import numpy as np
import warnings

# Suppress TensorFlow/Keras warnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras')

from src.evaluation.metrics import (
    GradientTracker,
    compute_accuracy,
    compute_success_rate,
    compare_approaches
)


class TestGradientTracker:
    """Test gradient tracking functionality."""
    
    def test_initialization(self):
        """Test gradient tracker initialization."""
        tracker = GradientTracker()
        
        assert tracker.gradient_norms == []
        assert hasattr(tracker, 'barren_plateau_threshold')
        assert tracker.barren_plateau_threshold == 1e-6
    
    def test_update_single_gradient(self):
        """Test updating with single gradient."""
        tracker = GradientTracker()
        
        # Create dummy gradient
        gradient = np.array([0.1, 0.2, 0.3])
        tracker.update([gradient])
        
        assert len(tracker.gradient_norms) == 1
        assert tracker.gradient_norms[0] > 0
    
    def test_update_multiple_gradients(self):
        """Test updating with multiple gradients."""
        tracker = GradientTracker()
        
        # Create multiple gradients
        gradients = [
            np.array([0.1, 0.2]),
            np.array([0.3, 0.4]),
            np.array([0.5, 0.6])
        ]
        tracker.update(gradients)
        
        # Each gradient adds one norm entry
        assert len(tracker.gradient_norms) == 3
    
    def test_update_multiple_steps(self):
        """Test multiple update steps."""
        tracker = GradientTracker()
        
        for i in range(5):
            gradient = np.random.rand(10)
            tracker.update([gradient])
        
        assert len(tracker.gradient_norms) == 5
    
    def test_get_statistics(self):
        """Test getting gradient statistics."""
        tracker = GradientTracker()
        
        # Add some gradients
        for _ in range(10):
            gradient = np.random.rand(5)
            tracker.update([gradient])
        
        stats = tracker.get_statistics()
        
        assert 'mean_norm' in stats
        assert 'std_norm' in stats
        assert 'variance' in stats  # Changed from 'mean_variance'
        assert 'min_norm' in stats
        assert 'max_norm' in stats
        assert 'median_norm' in stats
        assert 'total_updates' in stats
        
        assert stats['mean_norm'] > 0
        assert stats['std_norm'] >= 0
        assert stats['variance'] >= 0
    
    def test_detect_barren_plateau_no_plateau(self):
        """Test barren plateau detection with normal gradients."""
        tracker = GradientTracker(barren_plateau_threshold=1e-6)
        
        # Add normal-sized gradients
        for _ in range(10):
            gradient = np.random.rand(5) * 0.1  # ~0.1 magnitude
            tracker.update([gradient])
        
        has_plateau = tracker.detect_barren_plateau()  # No threshold parameter
        assert has_plateau is False
    
    def test_detect_barren_plateau_with_plateau(self):
        """Test barren plateau detection with vanishing gradients."""
        tracker = GradientTracker(barren_plateau_threshold=1e-6)
        
        # Add very small gradients
        for _ in range(10):
            gradient = np.random.rand(5) * 1e-8  # Very small
            tracker.update([gradient])
        
        has_plateau = tracker.detect_barren_plateau()  # No threshold parameter
        assert has_plateau is True
    
    def test_detect_barren_plateau_custom_threshold(self):
        """Test barren plateau detection with custom threshold."""
        # Use custom threshold at initialization
        tracker_low = GradientTracker(barren_plateau_threshold=1e-6)
        tracker_high = GradientTracker(barren_plateau_threshold=1e-4)
        
        # Add gradients around threshold
        for _ in range(10):
            gradient = np.random.rand(5) * 1e-5
            tracker_low.update([gradient])
            tracker_high.update([gradient])
        
        # Should detect with high threshold
        assert tracker_high.detect_barren_plateau() is True
        
        # Should not detect with low threshold
        assert tracker_low.detect_barren_plateau() is False
    
    def test_empty_tracker(self):
        """Test statistics on empty tracker."""
        tracker = GradientTracker()
        
        stats = tracker.get_statistics()
        
        # Should return empty dict when no gradients
        assert stats == {}
    
    def test_get_variance_trajectory(self):
        """Test variance trajectory computation."""
        tracker = GradientTracker()
        
        # Need at least window_size gradients
        for _ in range(60):
            gradient = np.random.rand(5) * 0.1
            tracker.update([gradient])
        
        trajectory = tracker.get_variance_trajectory(window_size=50)
        assert len(trajectory) > 0
        assert all(isinstance(v, float) for v in trajectory)


class TestAccuracyMetrics:
    """Test accuracy computation functions."""
    
    def test_compute_accuracy_perfect(self):
        """Test accuracy computation with perfect predictions."""
        # Labels should be 0/1, not -1/1
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([0.9, 0.1, 0.8, 0.2, 0.95])  # Probabilities > 0.5 → class 1
        
        accuracy = compute_accuracy(y_true, y_pred)
        assert accuracy == 100.0
    
    def test_compute_accuracy_half(self):
        """Test accuracy with 50% correct predictions."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0.9, 0.3, 0.7, 0.2])  # [1, 0, 1, 0] vs [1, 1, 0, 0] = 50%
        
        accuracy = compute_accuracy(y_true, y_pred)
        assert accuracy == 50.0
    
    def test_compute_accuracy_zero(self):
        """Test accuracy with all wrong predictions."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])  # All < 0.5 → class 0
        
        accuracy = compute_accuracy(y_true, y_pred)
        assert accuracy == 0.0
    
    def test_compute_accuracy_continuous_predictions(self):
        """Test accuracy with continuous predictions."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.8, 0.1, 0.7, 0.3])  # [1, 0, 1, 0] → 100%
        
        accuracy = compute_accuracy(y_true, y_pred)
        assert accuracy == 100.0
    
    def test_compute_accuracy_edge_cases(self):
        """Test accuracy with edge case predictions."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.51, 0.49, 0.6, 0.4])  # [1, 0, 1, 0] → 100%
        
        accuracy = compute_accuracy(y_true, y_pred)
        assert accuracy == 100.0
    
    def test_compute_accuracy_mismatched_length(self):
        """Test error handling for mismatched lengths."""
        y_true = np.array([1, 0, 1])
        y_pred = np.array([0.8, 0.2])
        
        # Will raise ValueError in comparison due to shape mismatch
        with pytest.raises((ValueError, AssertionError, IndexError)):
            compute_accuracy(y_true, y_pred)


class TestSuccessRate:
    """Test success rate computation."""
    
    def test_compute_success_rate_all_success(self):
        """Test success rate with all runs successful."""
        accuracies = [90.0, 92.0, 95.0, 88.0, 91.0]
        threshold = 85.0
        
        success_rate = compute_success_rate(accuracies, threshold)
        assert success_rate == 100.0
    
    def test_compute_success_rate_partial(self):
        """Test success rate with partial success."""
        accuracies = [90.0, 80.0, 95.0, 75.0, 88.0]
        threshold = 85.0
        
        success_rate = compute_success_rate(accuracies, threshold)
        assert success_rate == 60.0  # 3 out of 5
    
    def test_compute_success_rate_no_success(self):
        """Test success rate with no successful runs."""
        accuracies = [70.0, 75.0, 80.0, 72.0, 78.0]
        threshold = 85.0
        
        success_rate = compute_success_rate(accuracies, threshold)
        assert success_rate == 0.0
    
    def test_compute_success_rate_threshold(self):
        """Test success rate with values at threshold."""
        accuracies = [85.0, 85.0, 84.9, 85.1]
        threshold = 85.0
        
        success_rate = compute_success_rate(accuracies, threshold)
        # 85.0 and above should count as success
        assert success_rate == 75.0  # 3 out of 4
    
    def test_compute_success_rate_empty(self):
        """Test success rate with empty list."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            compute_success_rate([], threshold=85.0)


class TestCompareApproaches:
    """Test approach comparison functionality."""
    
    def test_compare_approaches_basic(self):
        """Test basic approach comparison."""
        # Structure matches what compare_approaches expects
        results = {
            'baseline': {
                'test_acc': 85.0,
                'training_time': 100.0,
                'gradient_stats': {'mean_norm': 0.001},
                'barren_plateau_detected': False
            },
            'layerwise': {
                'test_acc': 90.0,
                'training_time': 120.0,
                'gradient_stats': {'mean_norm': 0.002},
                'barren_plateau_detected': False
            }
        }
        
        comparison = compare_approaches(results)
        
        assert 'baseline' in comparison
        assert 'layerwise' in comparison
        assert 'summary' in comparison
        assert 'test_accuracy' in comparison['baseline']
        assert 'training_time' in comparison['baseline']
    
    def test_compare_approaches_statistics(self):
        """Test statistical computations in comparison."""
        results = {
            'approach1': {
                'test_acc': 85.0,
                'training_time': 105.0,
                'gradient_stats': {'mean_norm': 0.001},
                'barren_plateau_detected': False
            }
        }
        
        comparison = compare_approaches(results)
        
        # Check structure
        assert 'test_accuracy' in comparison['approach1']
        assert comparison['approach1']['test_accuracy'] == 85.0
    
    def test_compare_approaches_multiple(self):
        """Test comparison with multiple approaches."""
        results = {
            'baseline': {
                'test_acc': 81.0,
                'training_time': 101.0,
                'gradient_stats': {'mean_norm': 0.001},
                'barren_plateau_detected': False
            },
            'layerwise': {
                'test_acc': 86.0,
                'training_time': 121.0,
                'gradient_stats': {'mean_norm': 0.002},
                'barren_plateau_detected': False
            },
            'local_cost': {
                'test_acc': 89.0,
                'training_time': 96.0,
                'gradient_stats': {'mean_norm': 0.0015},
                'barren_plateau_detected': False
            }
        }
        
        comparison = compare_approaches(results)
        
        assert 'baseline' in comparison
        assert 'layerwise' in comparison
        assert 'local_cost' in comparison
        assert 'summary' in comparison


class TestMetricsIntegration:
    """Integration tests for metrics."""
    
    def test_full_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        # Initialize tracker
        tracker = GradientTracker(barren_plateau_threshold=1e-5)
        
        # Simulate training with gradients
        for epoch in range(20):
            gradient = np.random.rand(10) * (0.1 / (epoch + 1))  # Decreasing
            tracker.update([gradient])
        
        # Get statistics
        stats = tracker.get_statistics()
        assert stats['mean_norm'] > 0
        
        # Check for barren plateau (no threshold parameter)
        has_plateau = tracker.detect_barren_plateau()
        assert isinstance(has_plateau, bool)
        
        # Compute accuracies (use 0/1 labels)
        y_true = np.array([1, 0, 1, 0] * 5)
        y_pred = np.array([0.9, 0.1, 0.8, 0.2] * 5)
        accuracy = compute_accuracy(y_true, y_pred)
        assert accuracy == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])