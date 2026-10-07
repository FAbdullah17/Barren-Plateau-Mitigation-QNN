"""Aggregation and paired comparison of experiment results.

Aggregates the ``metrics.json`` files produced by the three runners (one file
per seed-triple under ``results/<approach>/depth_<L>/seed_<N>/``) and runs the
comparison statistics:

* per (approach, depth) summary: mean, SD, bootstrap 95% CI of test accuracy;
* per depth, pairwise **paired** comparisons across approaches (same seed
  triples): paired t-test, Wilcoxon, Cohen's d, Holm-Bonferroni correction.

There is no "fabricated data" path: a missing or malformed result fails loudly
so a partial/invalid experiment cannot masquerade as a completed run.

Usage:
    python experiments/run_comparison.py --results-dir results \\
        --depths 4 6 8 --output results/comparison
"""

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from src.evaluation import plot_comparison, plot_gradient_trajectory

# Required top-level fields for a valid per-run metrics.json.
REQUIRED_METRICS_FIELDS = [
    'config', 'data_seed', 'init_seed', 'training_seed', 'seed_index',
    'test_acc', 'test_loss', 'training_time_seconds', 'total_updates',
    'n_parameters', 'training_diagnostic', 'history', 'pca_info',
]

DEFAULT_BOOTSTRAP = 2000


def _jsonify(obj):
    """Recursively convert numpy types to JSON-native Python types."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _load_all(results_dir: Path, approaches, depths):
    """Load every metrics.json; fail loudly on missing/malformed results.

    Returns:
        dict[(approach, depth)] -> {seed_index: metrics_dict}
    """
    data = {}
    for approach in approaches:
        for depth in depths:
            key = (approach, depth)
            exp_dir = results_dir / approach / f'depth_{depth}'
            if not exp_dir.exists():
                raise FileNotFoundError(
                    f"No results directory for {approach} depth {depth}: {exp_dir}"
                )
            found = {}
            for metrics_file in sorted(exp_dir.glob(f'seed_*/metrics.json')):
                try:
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                except (json.JSONDecodeError, OSError) as exc:
                    raise RuntimeError(
                        f"Unreadable metrics.json for {approach} depth {depth}: "
                        f"{metrics_file} ({exc})"
                    ) from exc
                missing = [k for k in REQUIRED_METRICS_FIELDS if k not in metrics]
                if missing:
                    raise ValueError(
                        f"metrics.json {metrics_file} is missing required "
                        f"fields: {missing}. Refusing to analyze partial results."
                    )
                seed_index = metrics['seed_index']
                if seed_index in found:
                    raise ValueError(
                        f"Duplicate seed_index {seed_index} for {approach} "
                        f"depth {depth}: {metrics_file}"
                    )
                found[seed_index] = metrics
            if not found:
                raise RuntimeError(
                    f"No completed runs for {approach} depth {depth} in {exp_dir}."
                )
            data[key] = found
    return data


def _bootstrap_ci(values: np.ndarray, n_boot: int, seed: int = 0):
    """Percentile (Monte-Carlo) 95% CI of the mean via bootstrap resampling."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _summarize_group(metrics_by_seed):
    """Per (approach, depth) summary of the primary outcomes."""
    keys = ['test_acc', 'test_loss', 'training_time_seconds',
            'mean_param_grad_variance']
    summary = {}
    for metric in keys:
        if metric == 'mean_param_grad_variance':
            values = np.asarray(
                [m['training_diagnostic']['mean_param_grad_variance']
                 for m in metrics_by_seed.values()]
            )
        else:
            values = np.asarray([m[metric] for m in metrics_by_seed.values()])
        entry = {
            'mean': float(values.mean()),
            'sd': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
        if metric == 'test_acc':
            entry['ci_95'] = _bootstrap_ci(values, DEFAULT_BOOTSTRAP)
        summary[metric] = entry
    summary['n'] = len(metrics_by_seed)
    return summary


def _aligned_pair(a: dict, b: dict):
    """Test-accuracy arrays for common seed indices, plus the index list."""
    common = sorted(set(a) & set(b))
    xa = np.asarray([a[i]['test_acc'] for i in common])
    xb = np.asarray([b[i]['test_acc'] for i in common])
    return xa, xb, common


def _paired_stats(xa: np.ndarray, xb: np.ndarray):
    """Paired comparison statistics."""
    diff = xa - xb
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        t_stat, t_p = stats.ttest_rel(xa, xb)
        w_stat, w_p = stats.wilcoxon(diff)
    sd_diff = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    cohens_d = float(diff.mean() / sd_diff) if sd_diff > 0 else 0.0
    return {
        'n': int(len(diff)),
        'mean_diff': float(diff.mean()),
        'sd_diff': sd_diff,
        'cohens_d': cohens_d,
        'ttest_t': float(t_stat),
        'ttest_p': float(t_p),
        'wilcoxon_stat': float(w_stat),
        'wilcoxon_p': float(w_p),
    }


def _holm_decision(pairs):
    """Holm-Bonferroni corrected rejection decisions (alpha in meta)."""
    k = len(pairs)
    order = np.argsort([p['ttest_p'] for p in pairs])
    rejected = [False] * k
    for rank, idx in enumerate(order, start=1):
        if pairs[idx]['ttest_p'] <= 0.05 / (k - rank + 1):
            rejected[idx] = True
        else:
            break
    return rejected


def _report_missing(pairwise_per_depth, data, alpha):
    """Report seed sets that are not identical across approaches (warn only)."""
    approaches = sorted({a for (a, _) in data})
    for depth in sorted(pairwise_per_depth):
        per_approach = {
            approach: set(data[(approach, depth)])
            for approach in approaches
            if (approach, depth) in data
        }
        if not per_approach:
            continue
        reference = max(per_approach.values(), key=len)
        for approach, seeds in per_approach.items():
            if seeds != reference:
                print(
                    f"WARNING: {approach} depth {depth} has seed indices "
                    f"{sorted(seeds)}; reference {sorted(reference)}. "
                    f"Paired tests use the intersection."
                )


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate results and run the paired analysis'
    )
    parser.add_argument(
        '--results-dir', type=str, default='results',
        help='Directory containing <approach>/depth_<L>/seed_<N>/metrics.json',
    )
    parser.add_argument('--depths', type=int, nargs='+', default=[4, 6, 8])
    parser.add_argument(
        '--approaches', type=str, nargs='+',
        default=['baseline', 'layerwise', 'local_cost'],
    )
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--output', type=str, default='results/comparison')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        parser.error(f"--results-dir not found: {results_dir}")

    data = _load_all(results_dir, args.approaches, args.depths)

    # Per (approach, depth) summary.
    per_depth = {}
    for approach in args.approaches:
        for depth in args.depths:
            per_depth.setdefault(int(depth), {})[approach] = _summarize_group(
                data[(approach, depth)]
            )

    # Pooled across depths per approach.
    pooled = {}
    for approach in args.approaches:
        combined = {}
        for depth in args.depths:
            for seed_index, metrics in data[(approach, depth)].items():
                combined[(depth, seed_index)] = metrics
        pooled[approach] = _summarize_group(combined)

    # Pairwise paired comparisons per depth.
    pairwise_per_depth = {}
    for depth in args.depths:
        pairs = []
        names = args.approaches
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                xa, xb, common = _aligned_pair(
                    data[(a, depth)], data[(b, depth)]
                )
                if len(common) == 0:
                    raise RuntimeError(
                        f"No shared seed indices between {a} and {b} at depth {depth}."
                    )
                entry = _paired_stats(xa, xb)
                entry['approach_a'] = a
                entry['approach_b'] = b
                entry['seed_indices'] = common
                pairs.append(entry)
        for entry, rejected in zip(pairs, _holm_decision(pairs)):
            entry['holm_reject'] = rejected
        pairwise_per_depth[int(depth)] = pairs

    # Pooled paired comparison across all depths (same seed index per depth).
    pooled_pairs = []
    names = args.approaches
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            xa, xb = [], []
            common = None
            for depth in args.depths:
                ca, cb, idx = _aligned_pair(data[(a, depth)], data[(b, depth)])
                xa.append(ca)
                xb.append(cb)
                common = idx
            xa = np.concatenate(xa)
            xb = np.concatenate(xb)
            entry = _paired_stats(xa, xb)
            entry['approach_a'] = a
            entry['approach_b'] = b
            entry['seed_indices'] = common
            pooled_pairs.append(entry)
    for entry, rejected in zip(pooled_pairs, _holm_decision(pooled_pairs)):
        entry['holm_reject'] = rejected

    _report_missing(pairwise_per_depth, data, args.alpha)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'meta': {
            'results_dir': str(results_dir),
            'depths': [int(d) for d in args.depths],
            'approaches': list(args.approaches),
            'alpha': args.alpha,
            'multiple_comparison': 'holm',
            'generated_utc': datetime.now(timezone.utc).isoformat(),
        },
        'per_depth': per_depth,
        'pooled': pooled,
        'pairwise_per_depth': pairwise_per_depth,
        'pairwise_pooled': pooled_pairs,
    }
    with open(output_dir / 'comparison.json', 'w') as f:
        json.dump(report, f, indent=2, default=_jsonify)

    # CSV: per (approach, depth) summary.
    with open(output_dir / 'summary_statistics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['approach', 'depth', 'n', 'test_acc_mean',
                         'test_acc_sd', 'test_acc_ci_lo', 'test_acc_ci_hi'])
        for depth in args.depths:
            for approach in args.approaches:
                entry = per_depth[int(depth)][approach]
                acc = entry['test_acc']
                writer.writerow([
                    approach, depth, entry['n'],
                    f"{acc['mean']:.4f}", f"{acc['sd']:.4f}",
                    f"{acc['ci_95'][0]:.4f}", f"{acc['ci_95'][1]:.4f}",
                ])

    # CSV: pairwise comparisons.
    with open(output_dir / 'pairwise_comparisons.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['depth', 'approach_a', 'approach_b', 'n', 'mean_diff',
                         'cohens_d', 'ttest_p', 'wilcoxon_p', 'holm_reject'])
        for depth in args.depths:
            for entry in pairwise_per_depth[int(depth)]:
                writer.writerow([
                    depth, entry['approach_a'], entry['approach_b'], entry['n'],
                    f"{entry['mean_diff']:.4f}", f"{entry['cohens_d']:.4f}",
                    f"{entry['ttest_p']:.4e}", f"{entry['wilcoxon_p']:.4e}",
                    entry['holm_reject'],
                ])

    # Plots.
    accuracy_summary = {
        approach: {
            str(depth): {
                'test_acc': per_depth[int(depth)][approach]['test_acc']['mean'],
                'test_acc_sd': per_depth[int(depth)][approach]['test_acc']['sd'],
                'test_acc_ci': per_depth[int(depth)][approach]['test_acc']['ci_95'],
            }
            for depth in args.depths
        }
        for approach in args.approaches
    }
    plot_comparison(
        accuracy_summary,
        save_path=str(output_dir / 'comparison_accuracy.png'),
        show=False,
    )

    for depth in args.depths:
        trajectories = {}
        for approach in args.approaches:
            traj_by_step = {}
            for seed_index, metrics in data[(approach, depth)].items():
                for step, value in zip(
                    metrics['training_diagnostic']['trajectory']['step'],
                    metrics['training_diagnostic']['trajectory']['mean_param_grad_variance'],
                ):
                    traj_by_step.setdefault(step, []).append(value)
            steps = sorted(traj_by_step)
            trajectories[approach] = {
                'trajectory': {
                    'step': steps,
                    'mean_param_grad_variance': [
                        float(np.mean(traj_by_step[s])) for s in steps
                    ],
                }
            }
        plot_gradient_trajectory(
            trajectories,
            save_path=str(output_dir / f'gradient_trajectory_depth_{depth}.png'),
            title=f'Mean parameter-gradient variance trajectories (L={depth})',
            show=False,
        )

    # Final report.
    lines = []
    lines.append('=' * 70)
    lines.append('COMPARISON REPORT')
    lines.append('=' * 70)
    lines.append(f"Results dir: {results_dir}")
    lines.append(f"Depths: {args.depths}   Approaches: {args.approaches}")
    lines.append(f"alpha: {args.alpha}   Multiple-comparison: Holm-Bonferroni")
    lines.append('')
    lines.append('PER (APPROACH, DEPTH) TEST ACCURACY (mean +- SD, 95% CI)')
    for depth in args.depths:
        for approach in args.approaches:
            entry = per_depth[int(depth)][approach]
            acc = entry['test_acc']
            lines.append(
                f"  {approach:11s} L={depth}: "
                f"{acc['mean']:.4f} +/- {acc['sd']:.4f} "
                f"(95% CI [{acc['ci_95'][0]:.4f}, {acc['ci_95'][1]:.4f}], "
                f"n={entry['n']})"
            )
    lines.append('')
    lines.append('PAIRWISE PAIRED COMPARISONS (alpha=0.05, Holm corrected)')
    for depth in args.depths:
        for entry in pairwise_per_depth[int(depth)]:
            lines.append(
                f"  L={depth} {entry['approach_a']} vs {entry['approach_b']}: "
                f"mean_diff={entry['mean_diff']:+.4f}, cohens_d={entry['cohens_d']:.3f}, "
                f"ttest_p={entry['ttest_p']:.3e}, wilcoxon_p={entry['wilcoxon_p']:.3e}, "
                f"holm_reject={entry['holm_reject']}"
            )
    lines.append('')
    lines.append('POOLED ACROSS DEPTHS')
    for approach in args.approaches:
        entry = pooled[approach]
        acc = entry['test_acc']
        lines.append(
            f"  {approach:11s}: {acc['mean']:.4f} +/- {acc['sd']:.4f} "
            f"(95% CI [{acc['ci_95'][0]:.4f}, {acc['ci_95'][1]:.4f}], n={entry['n']})"
        )
    for entry in pooled_pairs:
        lines.append(
            f"  {entry['approach_a']} vs {entry['approach_b']} (pooled): "
            f"mean_diff={entry['mean_diff']:+.4f}, cohens_d={entry['cohens_d']:.3f}, "
            f"ttest_p={entry['ttest_p']:.3e}, holm_reject={entry['holm_reject']}"
        )
    lines.append('')
    lines.append(f"JSON: {output_dir / 'comparison.json'}")
    lines.append(f"Summary CSV: {output_dir / 'summary_statistics.csv'}")
    lines.append(f"Pairwise CSV: {output_dir / 'pairwise_comparisons.csv'}")
    lines.append('=' * 70)

    report_text = '\n'.join(lines)
    with open(output_dir / 'final_report.txt', 'w') as f:
        f.write(report_text + '\n')
    print(report_text)


if __name__ == '__main__':
    main()