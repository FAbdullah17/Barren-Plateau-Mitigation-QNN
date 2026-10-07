"""Controlled circuit-landscape analysis.

This module implements the *no-training* study of the quantum cost landscape.
It characterises the cost-operator gradients ``∂C(θ)/∂θ`` over ``R`` random
parameter vectors (Monte-Carlo draws), for a fixed input state and a given
cost operator ``M ∈ {global Z⊗…⊗Z, local (1/n)ΣZᵢ}``, and estimates how the
landscape statistic ``V̄`` scales with qubit count.

This is deliberately decoupled from the training diagnostic
(``GradientTracker``, over samples during training). The two statistics measure
different objects and are never equated.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from .metrics import landscape_variance

# Parameter distributions.
# (a) Training-initialisation range.
PARAM_INIT_MIN = -0.05
PARAM_INIT_MAX = 0.05
# (b) Uniform over the parameter range.
PARAM_UNIFORM_HALF_RANGE = float(np.pi)

SUPPORTED_DISTRIBUTIONS = ("init", "uniform")

_BOOTSTRAP_ITERATIONS = 2000
_BOOTSTRAP_SEED = 42


def draw_parameter_vectors(
    n_parameters: int,
    n_instances: int,
    distribution: str = "init",
    seed: int = 0,
) -> np.ndarray:
    """Draw ``n_instances`` random parameter vectors, one per instance.

    Args:
        n_parameters: ``P``, the number of trainable parameters per circuit.
        n_instances: ``R``, the Monte-Carlo sample size.
        distribution: ``"init"`` for the training-initialisation distribution
            (uniform over ``[PARAM_INIT_MIN, PARAM_INIT_MAX]``) or ``"uniform"``
            for the landscape-theory distribution (uniform over
            ``(-π, π)``). The two are reported separately and never mixed.
        seed: RNG seed for reproducibility.

    Returns:
        ``float32`` array of shape ``(R, P)``.
    """
    if n_parameters < 1:
        raise ValueError(f"n_parameters must be >= 1, got {n_parameters}")
    if n_instances < 1:
        raise ValueError(f"n_instances must be >= 1, got {n_instances}")
    if distribution not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(
            f"unsupported distribution {distribution!r}; "
            f"expected one of {SUPPORTED_DISTRIBUTIONS}"
        )

    rng = np.random.RandomState(seed)
    if distribution == "init":
        out = rng.uniform(PARAM_INIT_MIN, PARAM_INIT_MAX, size=(n_instances, n_parameters))
    else:
        out = rng.uniform(
            -PARAM_UNIFORM_HALF_RANGE, PARAM_UNIFORM_HALF_RANGE, size=(n_instances, n_parameters)
        )
    return out.astype(np.float32)


def summarize_landscape(
    grad_instances: Sequence,
    cost_values: Sequence,
    n_bootstrap: int = _BOOTSTRAP_ITERATIONS,
    seed: int = _BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
) -> Dict:
    """Summarise one (n, depth, cost, distribution) configuration.

    Combines the ``V̄`` landscape statistic with the remaining per-configuration
    quantities in the output schema: mean-abs-gradient and the variance of the
    cost ``Var_θ[C]``. All three quantities carry Monte-Carlo uncertainty
    (bootstrap SE + 95% percentile CI) computed by resampling the ``R``
    instances.

    Args:
        grad_instances: ``(R, P)`` array of analytic parameter-shift gradients
            ``g_j^(r) = ∂C(θ^(r))/∂θ_j``.
        cost_values: ``(R,)`` array of cost values ``C(θ^(r))``.
        n_bootstrap: Bootstrap resamples for the ``V̄`` CI and ``Var[C]`` SE.
        seed: RNG seed for bootstrap reproducibility.
        confidence_level: Coverage of the percentile CI.

    Returns:
        dict with ``Vbar`` {mean, se, ci_95, median}, ``std_over_j``,
        ``mean_abs_grad`` {mean, se}, ``VarC`` {mean, se}, ``variance_per_parameter``
        and the dimensions ``n_instances``/``n_parameters``/``n_bootstrap``.
    """
    g = np.asarray(grad_instances, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError(f"grad_instances must be 2-D (R, P), got shape {g.shape}")
    c = np.asarray(cost_values, dtype=np.float64)
    if c.ndim != 1 or c.shape[0] != g.shape[0]:
        raise ValueError(
            f"cost_values must be 1-D of length R={g.shape[0]}, got shape {c.shape}"
        )
    if not np.all(np.isfinite(g)) or not np.all(np.isfinite(c)):
        raise ValueError("grad_instances and cost_values must contain only finite values")

    r, p = g.shape

    lv = landscape_variance(
        g, n_bootstrap=n_bootstrap, seed=seed, confidence_level=confidence_level
    )
    v_j = np.asarray(lv["variance_per_parameter"], dtype=np.float64)

    # Mean-abs-gradient over all (instance, parameter) entries.
    mab_mean = float(np.mean(np.abs(g)))

    # Monte-Carlo uncertainty via bootstrap resampling of the R instances
    # (same scheme as landscape_variance).
    alpha = 1.0 - confidence_level
    q = [100.0 * alpha / 2, 100.0 * (1 - alpha / 2)]
    rng = np.random.RandomState(seed)
    boot_mab = np.empty(n_bootstrap, dtype=np.float64)
    boot_var = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.randint(0, r, size=r)
        boot_mab[i] = np.mean(np.abs(g[idx]))
        boot_var[i] = np.var(c[idx])
    mab_se = float(np.std(boot_mab))
    mab_ci = [float(x) for x in np.percentile(boot_mab, q)]
    var_c = float(np.var(c))
    var_c_se = float(np.std(boot_var))
    var_c_ci = [float(x) for x in np.percentile(boot_var, q)]

    return {
        "Vbar": {
            "mean": lv["Vbar"],
            "se": lv["se"],
            "ci_95": lv["ci"],
            "median": float(np.median(v_j)),
        },
        "std_over_j": float(np.std(v_j)),
        "mean_abs_grad": {"mean": mab_mean, "se": mab_se, "ci_95": mab_ci},
        "VarC": {"mean": var_c, "se": var_c_se, "ci_95": var_c_ci},
        "variance_per_parameter": v_j.tolist(),
        "n_instances": int(r),
        "n_parameters": int(p),
        "n_bootstrap": int(n_bootstrap),
    }


def fit_scaling(
    x_values: Sequence,
    vbar_values: Sequence,
    model: str = "exponential",
) -> Dict:
    """Fit a scaling model of the landscape statistic over the observed range.

    Fits ``log V̄`` by ordinary least squares on:

    * ``exponential``: ``log V̄ = a + b·x`` (with ``x = n``);
    * ``power_law``: ``log V̄ = a + b·log x`` (with ``x = n``).

    Reports the fitted intercept ``a`` and slope ``b`` (the empirical scaling
    exponent) with standard errors / 95% CIs, plus model-quality diagnostics
    (R², RMSE, AIC, BIC) used to compare the two models.

    Args:
        x_values: ``(K,)`` abscissae (qubit counts, or depths).
        vbar_values: ``(K,)`` landscape means ``V̄``.
        model: ``"exponential"`` or ``"power_law"``.

    Returns:
        dict with ``model``, ``a``, ``b``, ``se_b``, ``ci_b_95``, ``r2``,
        ``rmse``, ``aic``, ``bic``, ``n_points``.
    """
    x = np.asarray(x_values, dtype=np.float64)
    v = np.asarray(vbar_values, dtype=np.float64)
    if x.ndim != 1 or v.ndim != 1:
        raise ValueError("x_values and vbar_values must be 1-D")
    if x.shape[0] != v.shape[0] or x.shape[0] == 0:
        raise ValueError("x_values and vbar_values must have equal, non-zero length")

    if model == "exponential":
        X = x
    elif model == "power_law":
        if np.any(x <= 0):
            raise ValueError("power_law fit requires strictly positive x_values")
        X = np.log(x)
    else:
        raise ValueError(f"unsupported model {model!r}; expected 'exponential' or 'power_law'")

    mask = v > 0
    if np.sum(mask) < 3:
        raise ValueError(
            f"need at least 3 strictly positive V̄ points to fit, got {int(np.sum(mask))}"
        )
    x_fit, v_fit = x[mask], v[mask]
    X_fit = np.log(x_fit) if model == "power_law" else x_fit
    y = np.log(v_fit)
    n = len(y)

    a = np.vstack([np.ones(n), X_fit]).T
    coef, _, _, _ = np.linalg.lstsq(a, y, rcond=None)
    b, a0 = float(coef[1]), float(coef[0])
    yhat = a @ coef
    resid = y - yhat
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else (1.0 if sse == 0 else 0.0)
    rmse = float(np.sqrt(sse / n))

    k = 2
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)

    dof = n - k
    if dof > 0:
        sigma2 = sse / dof
        se_b = float(np.sqrt(sigma2 * np.linalg.inv(a.T @ a)[1, 1]))
    else:
        se_b = 0.0
    ci_b = [float(b - 1.96 * se_b), float(b + 1.96 * se_b)]

    return {
        "model": model,
        "a": a0,
        "b": b,
        "se_b": se_b,
        "ci_b_95": ci_b,
        "r2": r2,
        "rmse": rmse,
        "aic": aic,
        "bic": bic,
        "n_points": int(n),
    }


def plot_variance_scaling(
    configs: Sequence[Dict],
    output_path: str,
    x_axis: str = "n",
) -> None:
    """Plot ``log10(V̄)`` against the scaling axis.

    Args:
        configs: per-configuration result dicts (as produced by the landscape
            runner), each with ``n_qubits``, ``depth``, ``cost``,
            ``distribution`` and ``Vbar.mean``.
        output_path: destination PNG path.
        x_axis: ``"n"`` (one subplot per depth, x = qubit count) or
            ``"depth"`` (one subplot per qubit count, x = depth).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if x_axis == "n":
        group_key, x_key, x_label = "depth", "n_qubits", "n_qubits"
        title_fmt = "depth={}"
    elif x_axis == "depth":
        group_key, x_key, x_label = "n_qubits", "depth", "depth"
        title_fmt = "n_qubits={}"
    else:
        raise ValueError(f"unsupported x_axis {x_axis!r}; expected 'n' or 'depth'")

    groups = sorted({c[group_key] for c in configs})
    series = sorted({(c["cost"], c["distribution"]) for c in configs})
    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 5), squeeze=False)
    for ax, g in zip(axes[0], groups):
        for cost, dist in series:
            pts = [
                c for c in configs
                if c[group_key] == g and c["cost"] == cost and c["distribution"] == dist
            ]
            xs = [c[x_key] for c in pts]
            ys = [np.log10(c["Vbar"]["mean"]) for c in pts]
            ax.plot(xs, ys, marker="o", label=f"{cost} / {dist}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("log10(V̄)")
        ax.set_title(title_fmt.format(g))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)