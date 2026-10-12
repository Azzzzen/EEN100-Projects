"""Topic template: copy this folder to topics/<topic_name>/ and edit SETTINGS.

Example question: how does recovery change with the expected average degree d,
holding n, k and the within/between ratio p_in / p_out = 8 fixed?

Run from the repository root:
    python topics/_template/run.py
Outputs go to results/<topic_name>/ (ignored by Git).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spectral_project import make_config, run_experiment, save_results
from spectral_project.plotting import plot_diagnostics

TOPIC = Path(__file__).parent.name
OUTPUT = Path("results") / TOPIC

N, K, RATIO = 300, 3, 8.0
SIZES = [N // K + (i < N % K) for i in range(K)]


def settings():
    """Yield (x_value, config). x is the one quantity this topic varies."""
    for d in [2, 4, 6, 8, 12, 16, 24]:
        # Exact size-weighted mean degree; each node has only s - 1 peers in its block.
        n = sum(SIZES)
        denominator = sum(s * (RATIO * (s - 1) + n - s) for s in SIZES) / n
        p_out = d / denominator
        yield d, make_config(f"{TOPIC}_d{d}", SIZES, RATIO * p_out, p_out)


def main():
    configs, rows, summary = [], [], []
    for x, config in settings():
        run_rows, run_summary, example = run_experiment(config)
        configs.append(config)
        rows += [{"x": x, **row} for row in run_rows]
        summary += [{"x": x, **row} for row in run_summary]
        plot_diagnostics(example, OUTPUT / "diagnostics" / config["name"], config["name"])
    save_results(OUTPUT, configs, rows, summary)

    fig, ax = plt.subplots(figsize=(5, 3.5), layout="constrained")
    for method in configs[0]["methods"]:
        points = [row for row in summary if row["method"] == method]
        ax.errorbar([row["x"] for row in points],
                    [row["misclassification_rate_mean"] for row in points],
                    yerr=[row["misclassification_rate_sd"] or 0 for row in points],
                    marker="o", capsize=3, label=method)
    ax.axhline(1 - max(SIZES) / sum(SIZES), color="grey", linestyle="--", linewidth=1,
               label="all in one cluster")
    ax.set(xlabel="Expected average degree d", ylabel="Misclassification rate", title=TOPIC)
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.savefig(OUTPUT / "sweep.png", dpi=180)
    plt.close(fig)
    print(f"Saved {len(rows)} runs to {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
