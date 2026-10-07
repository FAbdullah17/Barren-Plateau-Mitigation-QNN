"""
Controlled circuit-landscape analysis.

Cheap, **no-training** study of the quantum cost landscape. For each
``(n_qubits, depth, cost, distribution)`` configuration it draws ``R = 200``
random parameter vectors, evaluates the cost ``C(θ) = ⟨ψ(θ)|M|ψ(θ)⟩`` and its
analytic parameter-shift gradient ``g(θ) = ∂C/∂θ`` on the fixed all-zero input
state, and summarises the landscape statistic ``V̄`` with its uncertainty.

Writes ``variance_scaling.json``, per-configuration ``V_j`` ``.npz`` files and
the ``variance_vs_n.png`` / ``variance_vs_depth.png`` plots.

Run it (this script never runs itself):
    python experiments/run_variance_scaling.py                         # n ∈ {4,6,8}
    python experiments/run_variance_scaling.py --n 4 6 8 10            # n=10 if feasible
    python experiments/run_variance_scaling.py --r 20 --n-bootstrap 500  # quick smoke test
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Must precede any tensorflow import (legacy Keras, see tests/conftest.py).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cirq
import tensorflow as tf
import tensorflow_quantum as tfq

from src.models.quantum_circuit import QuantumCircuit, create_readout_operators
from src.evaluation.variance_scaling import (
    draw_parameter_vectors,
    fit_scaling,
    plot_variance_scaling,
    summarize_landscape,
)

ANSATZ_NOTE = "RY/RZ per qubit + linear CNOT chain (QuantumCircuit); fixed all-zero input state"


def _to_py(obj):
    """Recursively convert numpy scalars/arrays to plain Python for JSON."""
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def build_cost_model(n_qubits: int, n_layers: int, cost: str) -> tf.keras.Model:
    """Model outputting the raw cost expectation ``⟨ψ(θ)|M|ψ(θ)⟩`` (no data encoding)."""
    circuit = QuantumCircuit(n_qubits=n_qubits, n_layers=n_layers).get_circuit()
    readout = create_readout_operators(n_qubits, local=(cost == "local"))
    inputs = tf.keras.Input(shape=(), dtype=tf.string, name="circuits")
    # Kernel is overwritten via set_weights before any measurement.
    initializer = tf.keras.initializers.RandomUniform(0.0, 2 * np.pi, seed=0)
    pqc = tfq.layers.PQC(
        circuit,
        readout,
        differentiator=tfq.differentiators.ParameterShift(),
        initializer=initializer,
    )
    return tf.keras.Model(inputs=inputs, outputs=pqc(inputs))


def measure_cost_gradient(model: tf.keras.Model, empty_batch):
    """Returns ``measure(theta) -> (cost, grad)`` with parameter-shift gradients."""
    model(empty_batch, training=False)  # build the kernel once

    def measure(theta):
        model.set_weights([theta])
        with tf.GradientTape() as tape:
            out = model(empty_batch, training=False)  # (1, 1)
        # Differentiate the full output (TFQ drops gradients on reduced targets).
        grad = tape.gradient(out, model.trainable_variables[0])
        if grad is None:
            raise RuntimeError("gradient of cost wrt parameters is None")
        return float(out.numpy()[0, 0]), grad.numpy()

    return measure


def run_configuration(n, depth, cost, distribution, r, n_bootstrap, seed):
    model = build_cost_model(n, depth, cost)
    empty_batch = tfq.convert_to_tensor([cirq.Circuit()])
    measure = measure_cost_gradient(model, empty_batch)
    p = int(np.prod(model.trainable_variables[0].shape))  # runtime-derived

    thetas = draw_parameter_vectors(p, r, distribution, seed=seed)
    costs = np.empty(r, dtype=np.float64)
    grads = np.empty((r, p), dtype=np.float64)
    for i in range(r):
        costs[i], grads[i] = measure(thetas[i])

    summary = summarize_landscape(grads, costs, n_bootstrap=n_bootstrap, seed=seed)
    return summary, p


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", nargs="+", type=int, default=[4, 6, 8],
                        help="qubit counts (add 10 only if computationally feasible)")
    parser.add_argument("--depths", nargs="+", type=int, default=[4, 6, 8],
                        help="circuit depths L")
    parser.add_argument("--costs", nargs="+", default=["global", "local"],
                        help="cost operators")
    parser.add_argument("--distributions", nargs="+", default=["init", "uniform"],
                        help="parameter distributions (init, uniform)")
    parser.add_argument("--r", type=int, default=200, help="Monte-Carlo instances R")
    parser.add_argument("--n-bootstrap", type=int, default=2000,
                        help="bootstrap resamples for CIs/SEs")
    parser.add_argument("--seed", type=int, default=42, help="master RNG seed")
    parser.add_argument("--output", default="results/variance_scaling",
                        help="output directory")
    args = parser.parse_args(argv)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = []
    total = len(args.n) * len(args.depths) * len(args.costs) * len(args.distributions)
    started = time.time()
    idx = 0
    for n in args.n:
        for depth in args.depths:
            for cost in args.costs:
                for distribution in args.distributions:
                    idx += 1
                    t0 = time.time()
                    summary, p = run_configuration(
                        n, depth, cost, distribution, args.r, args.n_bootstrap, args.seed
                    )
                    elapsed = time.time() - t0
                    configs.append({
                        "n_qubits": int(n),
                        "depth": int(depth),
                        "cost": cost,
                        "distribution": distribution,
                        "R": int(args.r),
                        "P": int(p),
                        "Vbar": {k: _to_py(v) for k, v in summary["Vbar"].items()},
                        "std_over_j": _to_py(summary["std_over_j"]),
                        "mean_abs_grad": _to_py(summary["mean_abs_grad"]),
                        "VarC": _to_py(summary["VarC"]),
                        "Vj_npz": f"Vj_n{n}_d{depth}_{cost}_{distribution}.npz",
                    })
                    np.savez(
                        out_dir / configs[-1]["Vj_npz"],
                        Vj=np.asarray(summary["variance_per_parameter"], dtype=np.float64),
                    )
                    print(
                        f"[{idx}/{total}] n={n} depth={depth} {cost}/{distribution} "
                        f"V̄={summary['Vbar']['mean']:.3e} ({elapsed:.1f}s)"
                    )

    fits = []
    for cost in args.costs:
        for distribution in args.distributions:
            for depth in args.depths:
                pts = [c for c in configs
                       if c["cost"] == cost and c["distribution"] == distribution
                       and c["depth"] == depth]
                xs = [c["n_qubits"] for c in pts]
                vs = [c["Vbar"]["mean"] for c in pts]
                for model in ("exponential", "power_law"):
                    try:
                        f = fit_scaling(xs, vs, model=model)
                    except ValueError:
                        continue
                    fits.append({"axis": "n", "cost": cost, "distribution": distribution,
                                 "depth": depth, **f})
            for n in args.n:
                pts = [c for c in configs
                       if c["cost"] == cost and c["distribution"] == distribution
                       and c["n_qubits"] == n]
                xs = [c["depth"] for c in pts]
                vs = [c["Vbar"]["mean"] for c in pts]
                for model in ("exponential", "power_law"):
                    try:
                        f = fit_scaling(xs, vs, model=model)
                    except ValueError:
                        continue
                    fits.append({"axis": "depth", "cost": cost, "distribution": distribution,
                                 "n_qubits": n, **f})

    meta = {
        "ansatz": ANSATZ_NOTE,
        "r_instances": args.r,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "n_qubits": list(args.n),
        "depths": list(args.depths),
        "costs": list(args.costs),
        "distributions": list(args.distributions),
        "param_init_range": [-0.05, 0.05],
        "uniform_param_range": [-np.pi, np.pi],
        "elapsed_seconds": round(time.time() - started, 1),
    }
    payload = {"meta": meta, "configs": configs, "fits": fits}
    with open(out_dir / "variance_scaling.json", "w") as f:
        json.dump(_to_py(payload), f, indent=2)

    plot_variance_scaling(configs, out_dir / "variance_vs_n.png", x_axis="n")
    plot_variance_scaling(configs, out_dir / "variance_vs_depth.png", x_axis="depth")

    print(f"\nWrote {out_dir / 'variance_scaling.json'}")
    print(f"Wrote {out_dir / 'variance_vs_n.png'} and {out_dir / 'variance_vs_depth.png'}")
    print(f"Done in {meta['elapsed_seconds']}s")


if __name__ == "__main__":
    main()