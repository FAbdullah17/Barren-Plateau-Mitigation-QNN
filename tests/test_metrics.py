"""Unit tests for evaluation metrics."""

import pytest
import numpy as np
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
        assert tracker.gradient_variances == []
    
    def test_update_single_gradient(self):
        """Test updating with single gradient."""
        tracker = GradientTracker()
        
        # Create dummy gradient
        gradient = np.array([0.1, 0.2, 0.3])
        tracker.update([gradient])
        
        assert len(tracker.gradient_norms) == 1
        assert len(tracker.gradient_variances) == 1
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
        
        assert len(tracker.gradient_norms) == 1
        assert len(tracker.gradient_variances) == 1
    
    def test_update_multiple_steps(self):
        """Test multiple update steps."""
        tracker = GradientTracker()
        
        for i in range(5):
            gradient = np.random.rand(10)
            tracker.update([gradient])
        
        assert len(tracker.gradient_norms) == 5
        assert len(tracker.gradient_variances) == 5
    
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
        assert 'mean_variance' in stats
        assert 'std_variance' in stats
        
        assert stats['mean_norm'] > 0
        assert stats['std_norm'] >= 0
    
    def test_detect_barren_plateau_no_plateau(self):
        """Test barren plateau detection with normal gradients."""
        tracker = GradientTracker()
        
        # Add normal-sized gradients
        for _ in range(10):
            gradient = np.random.rand(5) * 0.1  # ~0.1 magnitude
            tracker.update([gradient])
        
        has_plateau = tracker.detect_barren_plateau(threshold=1e-6)
        assert has_plateau is False
    
    def test_detect_barren_plateau_with_plateau(self):
        """Test barren plateau detection with vanishing gradients."""
        tracker = GradientTracker()
        
        # Add very small gradients
        for _ in range(10):
            gradient = np.random.rand(5) * 1e-8  # Very small
            tracker.update([gradient])
        
        has_plateau = tracker.detect_barren_plateau(threshold=1e-6)
        assert has_plateau is True
    
    def test_detect_barren_plateau_custom_threshold(self):
        """Test barren plateau detection with custom threshold."""
        tracker = GradientTracker()
        
        # Add gradients around threshold
        for _ in range(10):
            gradient = np.random.rand(5) * 1e-5
            tracker.update([gradient])
        
        # Should detect with high threshold
        assert tracker.detect_barren_plateau(threshold=1e-4) is True
        
        # Should not detect with low threshold
        assert tracker.detect_barren_plateau(threshold=1e-6) is False
    
    def test_empty_tracker(self):
        """Test statistics on empty tracker."""
        tracker = GradientTracker()
        
        stats = tracker.get_statistics()
        
        # Should return NaN or handle gracefully
        assert 'mean_norm' in stats
        assert 'std_norm' in stats


class TestAccuracyMetrics:
    """Test accuracy computation functions."""
    
    def test_compute_accuracy_perfect(self):
        """Test accuracy computation with perfect predictions."""
        y_true = np.array([1, -1, 1, -1, 1])
        y_pred = np.array([1, -1, 1, -1, 1])
        
        accuracy = compute_accuracy(y_true, y_pred)
        assert accuracy == 100.0
    
    def test_compute_accuracy_half(self):
        """Test accuracy with 50% correct predictions."""
        y_true = np.array([1, 1, -1, -1])
        y_pred = np.array([1, -1, 1, -1])
        
        accuracy = compute_accuracy(y_true, y_pred)
        assert accuracy == 50.0
    
    def test_compute_accuracy_zero(self):
        """Test accuracy with all wrong predictions."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([-1, -1, -1, -1])
        
        accuracy = compute_accuracy(y_true, y_pred)
        assert accuracy == 0.0
    
    def test_compute_accuracy_continuous_predictions(self):
        """Test accuracy with continuous predictions."""
        y_true = np.array([1, -1, 1, -1])
        y_pred = np.array([0.8, -0.9, 0.7, -0.6])  # Should round to [1, -1, 1, -1]
        
        accuracy = compute_accuracy(y_true, y_pred)
        assert accuracy == 100.0
    
    def test_compute_accuracy_edge_cases(self):
        """Test accuracy with edge case predictions."""
        y_true = np.array([1, -1, 1, -1])
        y_pred = np.array([0.1, -0.1, 0.01, -0.01])  # Near zero
        
        # Should still classify correctly based on sign
        accuracy = compute_accuracy(y_true, y_pred)
        assert accuracy == 100.0
    
    def test_compute_accuracy_mismatched_length(self):
        """Test error handling for mismatched lengths."""
        y_true = np.array([1, -1, 1])
        y_pred = np.array([1, -1])
        
        with pytest.raises((ValueError, AssertionError)):
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
        results = {
            'baseline': {
                'final_accuracy': [85.0, 87.0, 86.0],
                'training_time': [100, 105, 102]
            },
            'layerwise': {
                'final_accuracy': [90.0, 92.0, 91.0],
                'training_time': [120, 125, 122]
            }
        }
        
        comparison = compare_approaches(results)
        
        assert 'baseline' in comparison
        assert 'layerwise' in comparison
        assert 'mean_accuracy' in comparison['baseline']
        assert 'std_accuracy' in comparison['baseline']
    
    def test_compare_approaches_statistics(self):
        """Test statistical computations in comparison."""
        results = {
            'approach1': {
                'final_accuracy': [80.0, 85.0, 90.0, 85.0, 80.0],
                'training_time': [100, 110, 105, 108, 102]
            }
        }
        
        comparison = compare_approaches(results)
        
        # Check mean is correct
        expected_mean = np.mean([80.0, 85.0, 90.0, 85.0, 80.0])
        assert abs(comparison['approach1']['mean_accuracy'] - expected_mean) < 0.01
        
        # Check std is computed
        assert comparison['approach1']['std_accuracy'] > 0
    
    def test_compare_approaches_multiple(self):
        """Test comparison with multiple approaches."""
        results = {
            'baseline': {'final_accuracy': [80, 82, 81], 'training_time': [100, 102, 101]},
            'layerwise': {'final_accuracy': [85, 87, 86], 'training_time': [120, 122, 121]},
            'local_cost': {'final_accuracy': [88, 90, 89], 'training_time': [95, 97, 96]}
        }
        
        comparison = compare_approaches(results)
        
        assert len(comparison) == 3
        assert all(k in comparison for k in ['baseline', 'layerwise', 'local_cost'])


class TestMetricsIntegration:
    """Integration tests for metrics."""
    
    def test_full_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        # Initialize tracker
        tracker = GradientTracker()
        
        # Simulate training with gradients
        for epoch in range(20):
            gradient = np.random.rand(10) * (0.1 / (epoch + 1))  # Decreasing
            tracker.update([gradient])
        
        # Get statistics
        stats = tracker.get_statistics()
        assert stats['mean_norm'] > 0
        
        # Check for barren plateau
        has_plateau = tracker.detect_barren_plateau(threshold=1e-5)
        assert isinstance(has_plateau, bool)
        
        # Compute accuracies
        y_true = np.array([1, -1, 1, -1] * 5)
        y_pred = np.array([1, -1, 1, -1] * 5)
        accuracy = compute_accuracy(y_true, y_pred)
        assert accuracy == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
