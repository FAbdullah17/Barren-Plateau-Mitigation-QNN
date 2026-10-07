"""Gradient-tracker tests.

Validates the two-statistic design:

* ``V̄^x`` (``GradientTracker``): per-parameter gradient variance over samples,
  mean/max absolute gradient, and the per-step trajectory — computed correctly
  on synthetic matrices.
* ``V̄`` (``landscape_variance``): variance over random parameter draws, SE and
  bootstrap CI — computed correctly and reproducibly.
"""

import pytest
import numpy as np

from src.evaluation.metrics import GradientTracker, landscape_variance


class TestGradientTrackerPartB:
    def test_metrics_correct_on_known_matrix(self):
        # (B=4, P=3): column variances over samples are 0, 0.25, 0.
        grad = np.array(
            [
                [0.0, 0.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 0.0, 2.0],
                [0.0, 1.0, 2.0],
            ]
        )
        tracker = GradientTracker()
        tracker.update(grad, step=7, samples=4)

        stats = tracker.get_statistics()
        assert stats['n_logged_steps'] == 1
        assert stats['mean_param_grad_variance'] == pytest.approx(
            (0.0 + 0.25 + 0.0) / 3.0
        )
        assert stats['mean_abs_grad'] == pytest.approx(np.mean(np.abs(grad)))
        assert stats['max_abs_grad'] == pytest.approx(2.0)
        assert stats['trajectory']['step'] == [7]
        assert stats['trajectory']['mean_param_grad_variance'] == pytest.approx(
            [(0.0 + 0.25 + 0.0) / 3.0]
        )

    def test_trajectory_aggregates_across_updates(self):
        tracker = GradientTracker()
        tracker.update(np.array([[1.0, 1.0], [-1.0, -1.0]]), step=0)
        # col0 var = 1, col1 var = 1 -> Vbar = 1.0
        tracker.update(np.array([[0.0, 2.0], [0.0, 2.0]]), step=5)
        # col0 var = 0, col1 var = 0 -> Vbar = 0.0

        stats = tracker.get_statistics()
        assert stats['n_logged_steps'] == 2
        assert stats['trajectory']['step'] == [0, 5]
        assert stats['mean_param_grad_variance'] == pytest.approx(0.5)
        assert stats['std_param_grad_variance'] == pytest.approx(0.5)
        # mean_abs over all entries: first update mean=1.0, second mean=1.0
        assert stats['mean_abs_grad'] == pytest.approx(1.0)
        assert stats['max_abs_grad'] == pytest.approx(2.0)

    def test_empty_tracker_statistics(self):
        stats = GradientTracker().get_statistics()
        assert stats['n_logged_steps'] == 0
        assert stats['trajectory']['step'] == []
        assert stats['trajectory']['mean_param_grad_variance'] == []
        assert stats['mean_param_grad_variance'] == 0.0

    def test_rejects_non_2d_or_nonfinite_input(self):
        tracker = GradientTracker()
        with pytest.raises(ValueError):
            tracker.update(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError):
            tracker.update(np.array([[1.0, np.nan], [2.0, 3.0]]))
        with pytest.raises(ValueError):
            tracker.update(np.empty((2, 0)))


class TestLandscapeVariancePartA:
    def test_known_matrix(self):
        # (R=4, P=3): per-parameter variances over instances are 0, 1, 1.25.
        instances = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 2.0, 1.0],
                [1.0, 0.0, 2.0],
                [1.0, 2.0, 3.0],
            ]
        )
        result = landscape_variance(instances, n_bootstrap=500, seed=0)

        assert result['n_instances'] == 4
        assert result['n_parameters'] == 3
        assert result['Vbar'] == pytest.approx((0.0 + 1.0 + 1.25) / 3.0)
        assert result['variance_per_parameter'] == pytest.approx([0.0, 1.0, 1.25])
        # MC bootstrap (resampling instances) must yield a finite, non-negative SE
        assert result['se'] >= 0.0
        lo, hi = result['ci']
        assert lo <= result['Vbar'] <= hi

    def test_bootstrap_reproducible(self):
        rng = np.random.RandomState(123)
        instances = rng.randn(8, 5)
        a = landscape_variance(instances, n_bootstrap=300, seed=7)
        b = landscape_variance(instances, n_bootstrap=300, seed=7)
        assert a['ci'] == b['ci']

    def test_constant_columns_give_zero_variance(self):
        instances = np.ones((5, 4))
        result = landscape_variance(instances, n_bootstrap=100)
        assert result['Vbar'] == 0.0
        assert result['se'] == 0.0

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            landscape_variance(np.array([1.0, 2.0]))
        with pytest.raises(ValueError):
            landscape_variance(np.zeros((3, 2)), n_bootstrap=0)
        with pytest.raises(ValueError):
            landscape_variance(np.zeros((3, 2)), confidence_level=1.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])