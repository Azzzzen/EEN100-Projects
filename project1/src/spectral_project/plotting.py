"""First-run diagnostics from saved arrays, without rerunning the experiment.

Figure contract: inspect whether a planted SBM produces an informative spectral
embedding. First-run adjacency is the context; method-specific spectra and raw
eigenvectors are diagnostics. This is a quantitative grid for course discussion,
not evidence of method superiority. Export a 10 x 6 inch PNG;
example.npz supplies all plotted data. Never select the best repeat.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .spectral import METHODS


def plot_diagnostics(example, output, name="baseline"):
    truth = example["truth"]
    order = np.argsort(truth, kind="stable")
    k = len(np.unique(truth))
    methods = [method for method in METHODS if f"{method}_eigenvalues" in example]
    names = {"unnormalized": r"$L = D - W$", "symmetric": r"$L_{\mathrm{sym}}$"}
    colors = {"unnormalized": "#527B99", "symmetric": "#B47963"}
    style = {"font.family": "DejaVu Sans", "font.size": 10,
             "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(10, 6), layout="constrained")
        grid = fig.add_gridspec(len(methods), 3, width_ratios=[1.15, 1, 1])
        graph_ax = fig.add_subplot(grid[:, 0])
        graph_ax.imshow(example["adjacency"][np.ix_(order, order)], cmap="Greys", vmin=0, vmax=1,
                        interpolation="nearest")
        graph_ax.set(title="a  Sampled adjacency", xlabel="Node (sorted by true group)", ylabel="Node")
        boundaries = np.cumsum(np.bincount(truth[order]))[:-1] - 0.5
        for boundary in boundaries:
            graph_ax.axhline(boundary, color="#B47963", linewidth=0.6)
            graph_ax.axvline(boundary, color="#B47963", linewidth=0.6)
        for row, method in enumerate(methods):
            spectrum_ax = fig.add_subplot(grid[row, 1])
            values = example[f"{method}_eigenvalues"]
            count = min(len(values), max(10, k + 2))
            spectrum_ax.plot(np.arange(1, count + 1), values[:count], "o-", markersize=4, color=colors[method])
            spectrum_ax.axvspan(k, k + 1, color=colors[method], alpha=0.15)
            spectrum_ax.set(title=f"{chr(98 + 2 * row)}  {names[method]} spectrum",
                            xlabel="Eigenvalue index", ylabel="Eigenvalue")
            spectrum_ax.set_xticks(np.arange(1, count + 1, max(1, count // 6)))
            vector_ax = fig.add_subplot(grid[row, 2])
            vectors = example[f"{method}_eigenvectors"][order].T
            limit = max(float(np.abs(vectors).max()), 1e-12)
            artist = vector_ax.imshow(vectors, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit,
                                      interpolation="nearest")
            vector_ax.set(title=f"{chr(99 + 2 * row)}  First {k} eigenvectors",
                          xlabel="Node (sorted by true group)", ylabel="Eigenvector index",
                          yticks=np.arange(k), yticklabels=np.arange(1, k + 1))
            fig.colorbar(artist, ax=vector_ax, label="Entry", fraction=0.05)
        fig.suptitle(f"{name} diagnostic | repeat 0 | n = {len(truth)}, k = {k}", fontsize=13)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        fig.savefig(output / "diagnostics.png", dpi=180)
        plt.close(fig)
