"""Controlled within/between contrast sweep; run from project1/.

Figure contract: test whether stronger community contrast improves recovery at
fixed expected degree. The main recovery plot shows misclassification rate; complete recovery is supplementary;
separate spectrum panels provide diagnostics, not a causal proof. Export PNG
at 10 x 7 inches for recovery; diagnostic figures use separate sizes. CSV files contain all plotted statistics.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spectral_project import make_config, run_experiment, save_results
from spectral_project.plotting import plot_diagnostics
from spectral_project.sbm import expected_average_degree

HERE = Path(__file__).resolve().parent
RATIOS = (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 12, 20)
BASELINE = json.loads((HERE.parents[1] / "configs/baseline.json").read_text())
SIZES = BASELINE["sizes"]
DEGREE = expected_average_degree(SIZES, BASELINE["block_probabilities"])
BASE_RATIO = BASELINE["block_probabilities"][0][0] / BASELINE["block_probabilities"][0][1]
REPEATS = BASELINE["repeats"]
MASTER_SEED = 2026092801
EXAMPLES = (BASE_RATIO,)  # Chosen before running; always use repeat 0.
METHODS = ("unnormalized", "symmetric")
LABELS = (r"$L$", r"$L_{\mathrm{sym}}$")
COLORS = ("#527B99", "#B47963")


def settings():
    children = np.random.SeedSequence(MASTER_SEED).spawn(len(RATIOS))
    n = sum(SIZES)
    for ratio, child in zip(RATIOS, children):
        denominator = sum(s * (ratio * (s - 1) + n - s) for s in SIZES) / n
        p_out = DEGREE / denominator
        config = make_config(f"contrast_r{ratio:g}", SIZES, ratio * p_out, p_out,
                             repeats=REPEATS, seed=BASELINE["seed"] if np.isclose(ratio, BASE_RATIO) else int(child.generate_state(1)[0]),
                             n_init=BASELINE["n_init"], methods=BASELINE["methods"])
        assert abs(expected_average_degree(SIZES, config["block_probabilities"]) - DEGREE) < 1e-10
        yield ratio, config


def wilson_interval(successes, total):
    """Pointwise 95% Wilson interval for a binomial success probability."""
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def draw_figures(summary, output):
    output.mkdir(parents=True, exist_ok=True)
    style = {"font.family": "DejaVu Sans", "font.size": 11,
             "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
        for method, label, color, marker in zip(METHODS, LABELS, COLORS, ('o', 's')):
            points = [row for row in summary if row['method'] == method]
            x = np.array([float(row['ratio']) for row in points])
            error = np.array([float(row['misclassification_rate_mean']) for row in points])
            sd = np.array([float(row['misclassification_rate_sd']) for row in points])
            ax.plot(x, error, marker=marker, color=color, label=label, markersize=4)
            ax.fill_between(x, np.maximum(0, error-sd), np.minimum(1, error+sd), color=color, alpha=.15)
        ax.axhline(1-max(SIZES)/sum(SIZES), color='#888888', ls=':', lw=1, label='All-one-group error')
        ax.axvline(BASE_RATIO, color='#777777', ls='--', lw=1, label=f'Baseline r={BASE_RATIO:g}')
        ax.set(xscale='log', xlabel='Within/between probability ratio', ylabel='Misclassification rate', ylim=(0, .72))
        ax.set_xticks([1,2,3,5,8,12,20], labels=[1,2,3,5,8,12,20]); ax.legend()
        fig.suptitle(f'Community recovery | n={sum(SIZES)}, sizes={SIZES}, expected degree={DEGREE:g}')
        fig.supxlabel(f'{REPEATS} graphs per setting | mean ± sample SD | lower error means more nodes recovered', fontsize=9)
        fig.savefig(output/'recovery.png', dpi=200); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout='constrained')
        for method, label, color in zip(METHODS, LABELS, COLORS):
            points = [row for row in summary if row['method'] == method]
            x = np.array([float(row['ratio']) for row in points])
            success = np.array([float(row['exact_recovery_rate']) for row in points])
            axes[0].plot(x, success, 'o-', color=color, label=label, markersize=4)
            axes[0].fill_between(x, [float(row['exact_ci_low']) for row in points],
                                 [float(row['exact_ci_high']) for row in points], color=color, alpha=.15)
            if method == METHODS[0]:
                for key, label2, marker in [('runs_with_isolates','At least one isolate','o'),
                                             ('runs_with_excess_components','More than k components','s')]:
                    axes[1].plot(x, [int(row[key])/int(row['repeats']) for row in points], marker+'-', label=label2)
        axes[0].set(title='Complete recovery (supplement)', ylabel='Fraction with zero errors')
        axes[1].set(title='Connectivity diagnostics', ylabel='Fraction of graphs')
        for ax in axes:
            ax.set(xscale='log', xlabel='Within/between probability ratio', ylim=(-.03,1.03))
            ax.set_xticks([1,2,3,5,8,12,20], labels=[1,2,3,5,8,12,20]); ax.legend()
        fig.supxlabel('Exact recovery ignores partial success; its bands are pointwise 95% Wilson intervals.', fontsize=9)
        fig.savefig(output/'recovery_supplement.png', dpi=180); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout='constrained')
        for ax, method, label, color in zip(axes, METHODS, LABELS, COLORS):
            points = [row for row in summary if row['method'] == method]
            ax.errorbar([float(row['ratio']) for row in points],
                        [float(row['eigengap_mean']) for row in points],
                        yerr=[float(row['eigengap_sd']) for row in points],
                        color=color, marker='o', capsize=3, markersize=4)
            ax.set(title=label, xlabel='Within/between probability ratio', ylabel=r'$\lambda_4 - \lambda_3$', xscale='log')
            ax.set_xticks([1,2,3,5,8,12,20], labels=[1,2,3,5,8,12,20])
        fig.suptitle(f'Spectral gaps | separate vertical scales; mean ± SD across {REPEATS} graphs')
        fig.savefig(output/'eigengap.png', dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--plot-only", action="store_true", help="Redraw figures from saved results")
    args = parser.parse_args()
    if args.plot_only:
        with (args.output / "overview" / "summary.csv").open() as handle:
            summary = list(csv.DictReader(handle))
    else:
        configs, rows, summary = [], [], []
        for ratio, config in settings():
            run_rows, run_summary, example = run_experiment(config)
            configs.append(config)
            parameters = {"ratio": ratio, "p_in": config["block_probabilities"][0][0],
                          "p_out": config["block_probabilities"][0][1]}
            rows.extend({**parameters, **row} for row in run_rows)
            for item in run_summary:
                successes = sum(row["exact_recovery"] for row in run_rows if row["method"] == item["method"])
                low, high = wilson_interval(successes, config["repeats"])
                summary.append({**parameters, **item, "exact_successes": successes,
                                "exact_ci_low": low, "exact_ci_high": high})
            if ratio in EXAMPLES:
                folder = args.output / f"r{ratio:g}"
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / "example.npz", **example)
            print(f"r={ratio:g}: " + "; ".join(
                f"{s['method']} error={s['misclassification_rate_mean']:.4f}, recovered={1-s['misclassification_rate_mean']:.1%}, ARI={s['ari_mean']:.3f}"
                for s in run_summary), flush=True)
        save_results(args.output / "overview", configs, rows, summary)
        metadata_path = args.output / "overview" / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.update(topic_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        embedding_script_sha256=hashlib.sha256((HERE / "embedding.py").read_bytes()).hexdigest(),
                        master_seed=MASTER_SEED, independent_seeds_across_ratios=True,
                        walkthrough_match=f"r={BASE_RATIO:g} uses baseline seed; repeat 0 matches the PPT",
                        examples={"ratios": list(EXAMPLES), "repeat": 0},
                        exact_interval="pointwise 95% Wilson; not simultaneous across ratios")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    draw_figures(summary, args.output / "overview")
    for ratio in EXAMPLES:
        folder = args.output / f"r{ratio:g}"
        with np.load(folder / "example.npz") as example:
            plot_diagnostics(example, folder, f"contrast r={ratio:g}")
    # Local import also works when this script is loaded by the unit tests.
    import importlib.util
    spec = importlib.util.spec_from_file_location("contrast_embedding", HERE / "embedding.py")
    embedding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(embedding)
    embedding.plot_embeddings(args.output, EXAMPLES)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
