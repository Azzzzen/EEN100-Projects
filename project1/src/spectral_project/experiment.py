"""Run one fixed SBM configuration; topic sweeps loop over configs in topics/."""

import csv
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .metrics import recovery_metrics
from .sbm import expected_average_degree, sample_sbm, validate_sbm
from .spectral import METHODS, spectral_cluster


def make_config(name, sizes, p_in, p_out, methods=METHODS, repeats=20, seed=20260928, n_init=20):
    """Planted-partition config: p_in on the diagonal, p_out everywhere else."""
    k = len(sizes)
    probabilities = [[p_in if i == j else p_out for j in range(k)] for i in range(k)]
    validated_sizes, probabilities_array = validate_sbm(sizes, probabilities)
    config = {"name": name, "sizes": validated_sizes.tolist(),
              "block_probabilities": probabilities_array.tolist(), "methods": list(methods),
              "repeats": repeats, "seed": seed, "n_init": n_init}
    validate_config(config)
    return config


def validate_config(config):
    required = {"name", "sizes", "block_probabilities", "methods", "repeats", "seed", "n_init"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"config must contain exactly these keys: {sorted(required)}")
    sizes, _ = validate_sbm(config["sizes"], config["block_probabilities"])
    if len(sizes) >= sizes.sum():
        raise ValueError("need more nodes than communities to compute the k-th eigengap")
    if not isinstance(config["name"], str) or not config["name"].strip():
        raise ValueError("name must be a nonempty string")
    for key in ("repeats", "n_init", "seed"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "seed" else 1):
            raise ValueError(f"{key} must be an integer >= {0 if key == 'seed' else 1}")
    methods = config["methods"]
    if (not isinstance(methods, list) or not methods
            or any(method not in METHODS for method in methods) or len(set(methods)) != len(methods)):
        raise ValueError(f"methods must be a nonempty list without duplicates, drawn from {METHODS}")


def run_experiment(config):
    """Return per-run rows, aggregate rows, and first-run diagnostic arrays.

    Each repeat has an independent graph seed and k-means seed. Both methods
    share these seeds; changing the method order never changes the sampled graph.
    """
    validate_config(config)
    n = sum(config["sizes"])
    k = len(config["sizes"])
    expected_degree = expected_average_degree(config["sizes"], config["block_probabilities"])
    rows, example = [], {}
    children = np.random.SeedSequence(config["seed"]).spawn(config["repeats"])
    for repeat, child in enumerate(children):
        graph_seed, kmeans_seed = map(int, child.generate_state(2))
        adjacency, truth = sample_sbm(config["sizes"], config["block_probabilities"], graph_seed)
        for method in config["methods"]:
            result = spectral_cluster(adjacency, k, method, seed=kmeans_seed, n_init=config["n_init"])
            rows.append({
                "name": config["name"], "repeat": repeat, "method": method, "graph_seed": graph_seed,
                "kmeans_seed": kmeans_seed, "n": n, "k": k,
                "expected_average_degree": expected_degree,
                "average_degree": float(adjacency.sum() / n),
                "isolated_nodes": result.isolated_nodes,
                "connected_components": result.connected_components,
                "clusters_found": result.clusters_found,
                "eigengap": result.eigengap,
                **recovery_metrics(truth, result.labels),
            })
            if repeat == 0:
                example[f"{method}_eigenvalues"] = result.eigenvalues
                example[f"{method}_eigenvectors"] = result.eigenvectors
                example[f"{method}_embedding"] = result.embedding
                example[f"{method}_labels"] = result.labels
        if repeat == 0:
            example.update(adjacency=adjacency, truth=truth)
    return rows, summarize(rows, config["methods"]), example


def summarize(rows, methods):
    summary = []
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        item = {"name": subset[0]["name"], "method": method, "repeats": len(subset)}
        for key in ("misclassification_rate", "ari", "eigengap", "average_degree"):
            values = [row[key] for row in subset]
            item[f"{key}_mean"] = float(np.mean(values))
            item[f"{key}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        item["exact_recovery_rate"] = float(np.mean([row["exact_recovery"] for row in subset]))
        item["runs_with_isolates"] = sum(row["isolated_nodes"] > 0 for row in subset)
        item["runs_with_excess_components"] = sum(row["connected_components"] > row["k"] for row in subset)
        item["runs_with_missing_clusters"] = sum(row["clusters_found"] < row["k"] for row in subset)
        summary.append(item)
    return summary


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_results(output, config, rows, summary, example=None):
    """Overwrite only known generated filenames in the selected output directory.

    `config` may be one config or a list of configs (a sweep); `example` is optional.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(config, indent=2, sort_keys=True) + "\n"
    (output / "config.json").write_text(serialized, encoding="utf-8")
    write_csv(output / "runs.csv", rows)
    write_csv(output / "summary.csv", summary)
    if example is not None:
        np.savez_compressed(output / "example.npz", **example)
    source_root = Path(__file__).parent
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in (
            "numpy", "scipy", "scikit-learn", "matplotlib", "threadpoolctl")},
        "config_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(source_root.glob("*.py"))},
        "example_repeat": 0 if example is not None else None,
        "sd_definition": "sample standard deviation across graphs (ddof=1); empty for one repeat",
        "note": "Finite simulation results, not a proof of asymptotic recovery. BLAS threads limited to 1.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
