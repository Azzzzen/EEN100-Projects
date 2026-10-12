"""Command-line baseline runner."""

import argparse
import json
from pathlib import Path

from .experiment import run_experiment, save_results


def main():
    parser = argparse.ArgumentParser(description="Run a reproducible SBM spectral clustering baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--output", type=Path, default=Path("results/baseline"))
    parser.add_argument("--no-plots", action="store_true", help="Save numerical results only")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        rows, summary, example = run_experiment(config)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    save_results(args.output, config, rows, summary, example)
    if not args.no_plots:
        from .plotting import plot_diagnostics
        plot_diagnostics(example, args.output, config["name"])
    for row in summary:
        print(f"{row['method']:14s}  error={row['misclassification_rate_mean']:.4f}  "
              f"ARI={row['ari_mean']:.4f}  exact={row['exact_recovery_rate']:.1%}  "
              f"repeats={row['repeats']}")
        if row["runs_with_isolates"] or row["runs_with_excess_components"] or row["runs_with_missing_clusters"]:
            print("  Inspect isolation/component/cluster-count diagnostics in runs.csv.")
    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
