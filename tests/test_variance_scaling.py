"""Landscape-scaling tests.

Validates the controlled circuit-landscape analysis building blocks:

* ``draw_parameter_vectors``: both parameter distributions, shapes and
  reproducibility;
* ``summarize_landscape``: the per-configuration statistic (``V̄`` with SE/CI,
  mean-abs-gradient, ``Var_θ[C]``) on known synthetic matrices;
* ``fit_scaling``: exponential vs power-law recovery, model diagnostics and
  error handling;
* ``plot_variance_scaling``: writes the expected PNG outputs.
"""

import numpy as np
import pytest

from src.evaluation.variance_scaling import (
    draw_parameter_vectors,
    fit_scaling,
    plot_variance_scaling,
    summarize_landscape,
    PARAM_INIT_MIN,
    PARAM_INIT_MAX,
    PARAM_UNIFORM_HALF_RANGE,
)


class TestDrawParameterVectors:
    def test_shapes_and_dtype(self):
        theta = draw_parameter_vectors(10, 5, "init", seed=1)
        assert theta.shape == (5, 10)
        assert theta.dtype == np.float32

    def test_init_range(self):
        theta = draw_parameter_vectors(50, 20, "init", seed=3)
        assert theta.min() >= PARAM_INIT_MIN - 1e-6
        assert theta.max() <= PARAM_INIT_MAX + 1e-6

    def test_uniform_range(self):
        theta = draw_parameter_vectors(50, 20, "uniform", seed=3)
        assert theta.min() >= -PARAM_UNIFORM_HALF_RANGE - 1e-6
        assert theta.max() <= PARAM_UNIFORM_HALF_RANGE + 1e-6
        # uniform over (-pi, pi) should actually reach out near the extremes
        assert theta.min() < -3.0
        assert theta.max() > 3.0

    def test_reproducible_and_seed_sensitive(self):
        a = draw_parameter_vectors(8, 4, "init", seed=7)
        b = draw_parameter_vectors(8, 4, "init", seed=7)
        c = draw_parameter_vectors(8, 4, "init", seed=8)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_invalid_distribution(self):
        with pytest.raises(ValueError):
            draw_parameter_vectors(4, 4, distribution="other", seed=0)

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            draw_parameter_vectors(0, 4, "init", seed=0)
        with pytest.raises(ValueError):
            draw_parameter_vectors(4, 0, "init", seed=0)


class TestSummarizeLandscape:
    def test_vbar_and_metrics_on_known_matrix(self):
        # (R=4, P=3): column variances 0, 0.25, 0 -> Vbar = 0.0833...
        grad = np.array(
            [
                [0.0, 0.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 0.0, 2.0],
                [0.0, 1.0, 2.0],
            ]
        )
        costs = np.array([0.0, 1.0, 2.0, 3.0])  # Var[C] = 1.25
        out = summarize_landscape(grad, costs, n_bootstrap=200, seed=0)

        assert out["n_instances"] == 4
        assert out["n_parameters"] == 3
        assert out["Vbar"]["mean"] == pytest.approx((0.0 + 0.25 + 0.0) / 3.0)
        assert out["Vbar"]["median"] == pytest.approx(0.0)
        assert out["Vbar"]["se"] >= 0.0
        lo, hi = out["Vbar"]["ci_95"]
        assert lo <= out["Vbar"]["mean"] <= hi
        # mean-abs-gradient over all entries; MC bootstrap SE and CI
        assert out["mean_abs_grad"]["mean"] == pytest.approx(np.mean(np.abs(grad)))
        assert out["mean_abs_grad"]["se"] >= 0.0
        mab_lo, mab_hi = out["mean_abs_grad"]["ci_95"]
        assert mab_lo <= out["mean_abs_grad"]["mean"] <= mab_hi
        # Var[C] over instances; MC bootstrap SE and CI
        assert out["VarC"]["mean"] == pytest.approx(1.25)
        assert out["VarC"]["se"] >= 0.0
        vc_lo, vc_hi = out["VarC"]["ci_95"]
        assert vc_lo <= 1.25 <= vc_hi
        assert out["std_over_j"] == pytest.approx(np.std([0.0, 0.25, 0.0]))

    def test_var_c_se_with_nonzero_instances(self):
        rng = np.random.RandomState(1)
        grad = rng.normal(size=(30, 4))
        costs = rng.normal(size=30)
        out = summarize_landscape(grad, costs, n_bootstrap=300, seed=5)
        assert out["VarC"]["se"] > 0.0
        # bootstrap resampling should be reproducible for a fixed seed
        again = summarize_landscape(grad, costs, n_bootstrap=300, seed=5)
        assert again["VarC"]["se"] == out["VarC"]["se"]
        assert again["Vbar"]["ci_95"] == out["Vbar"]["ci_95"]

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            summarize_landscape(np.zeros((4,)), np.zeros(4))
        with pytest.raises(ValueError):
            summarize_landscape(np.zeros((4, 3)), np.zeros(5))
        with pytest.raises(ValueError):
            summarize_landscape(np.array([[1.0, np.nan]]), np.array([0.0]))


class TestFitScaling:
    def test_exponential_recovery(self):
        n = np.array([4.0, 6.0, 8.0])
        true_b, true_a = -0.3, 1.0
        vbar = np.exp(true_a + true_b * n)
        out = fit_scaling(n, vbar, model="exponential")
        assert out["b"] == pytest.approx(true_b, rel=1e-6)
        assert out["a"] == pytest.approx(true_a, rel=1e-6)
        assert out["r2"] == pytest.approx(1.0)
        assert out["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert out["n_points"] == 3
        assert np.isfinite(out["aic"]) and np.isfinite(out["bic"])
        assert out["ci_b_95"][0] <= out["b"] <= out["ci_b_95"][1]

    def test_power_law_recovery(self):
        n = np.array([4.0, 6.0, 8.0])
        true_b, true_a = -1.7, 2.0
        vbar = np.exp(true_a + true_b * np.log(n))
        out = fit_scaling(n, vbar, model="power_law")
        assert out["b"] == pytest.approx(true_b, rel=1e-6)
        assert out["a"] == pytest.approx(true_a, rel=1e-6)
        assert out["r2"] == pytest.approx(1.0)

    def test_exponential_misses_power_law_trend(self):
        # A pure power-law dataset should be worse described by an exponential
        n = np.array([4.0, 6.0, 8.0])
        vbar = np.exp(2.0 - 1.7 * np.log(n))
        exp = fit_scaling(n, vbar, model="exponential")
        pl = fit_scaling(n, vbar, model="power_law")
        assert pl["r2"] > exp["r2"]

    def test_errors(self):
        with pytest.raises(ValueError):
            fit_scaling([4.0, 6.0], [0.1, 0.05], "exponential")  # <3 points
        with pytest.raises(ValueError):
            fit_scaling([4.0, 6.0, 8.0], [0.0, 0.0, 0.0], "exponential")  # non-positive
        with pytest.raises(ValueError):
            fit_scaling([0.0, 2.0, 4.0], [0.1, 0.05, 0.01], "power_law")  # x<=0
        with pytest.raises(ValueError):
            fit_scaling([4.0, 6.0, 8.0], [0.1, 0.05, 0.01], "cubic")


class TestPlotVarianceScaling:
    def test_writes_png(self, tmp_path):
        configs = [
            {"n_qubits": n, "depth": d, "cost": cost, "distribution": dist,
             "Vbar": {"mean": 1e-3 * n}}
            for n in (4, 6, 8)
            for d in (4,)
            for cost in ("global", "local")
            for dist in ("init", "uniform")
        ]
        out_n = tmp_path / "variance_vs_n.png"
        out_d = tmp_path / "variance_vs_depth.png"
        plot_variance_scaling(configs, str(out_n), x_axis="n")
        plot_variance_scaling(configs, str(out_d), x_axis="depth")
        assert out_n.exists() and out_n.stat().st_size > 0
        assert out_d.exists() and out_d.stat().st_size > 0

    def test_invalid_axis(self, tmp_path):
        with pytest.raises(ValueError):
            plot_variance_scaling([], str(tmp_path / "x.png"), x_axis="layers")