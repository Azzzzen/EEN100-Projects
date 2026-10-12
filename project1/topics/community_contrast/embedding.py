"""PNG scatter and node-link views of saved graphs in two spectral coordinates.

Figure contract: show how true and predicted groups occupy a common graph-derived
layout. Each ratio has its own figure; columns are alternative labelings of the
same nodes. The layout uses PCA of the full normalized Lsym embedding without consulting truth. It is a 2D
visualization, not either algorithm's full k-dimensional clustering input.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

PALETTE = np.array(["#4477AA", "#EE9944", "#228866"])


def align_for_display(truth, predicted):
    """Match predicted group IDs to truth for consistent colors, not positions."""
    groups, true_indices = np.unique(truth, return_inverse=True)
    clusters, predicted_indices = np.unique(predicted, return_inverse=True)
    counts = np.zeros((len(groups), len(clusters)), dtype=int)
    np.add.at(counts, (true_indices, predicted_indices), 1)
    rows, cols = linear_sum_assignment(-counts)
    mapping = {clusters[col]: groups[row] for row, col in zip(rows, cols)}
    return np.array([mapping.get(label, -1) for label in predicted])


def plot_embeddings(output, ratios):
    output = Path(output)
    style = {"font.family": "DejaVu Sans", "font.size": 10,
             "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(style):
        for ratio in ratios:
            fig = plt.figure(figsize=(11, 4.3), layout="constrained")
            grid = fig.add_gridspec(2, 3, height_ratios=[1, 0.2])
            axes = [fig.add_subplot(grid[0, col]) for col in range(3)]
            folder = output / f"r{ratio:g}"
            with np.load(folder / "example.npz") as example:
                truth = example["truth"]
                adjacency = example["adjacency"]
                pca = PCA(n_components=2, svd_solver="full")
                coordinates = pca.fit_transform(example["symmetric_embedding"])
                predictions = [align_for_display(truth, example[f"{method}_labels"])
                               for method in ("unnormalized", "symmetric")]
            if coordinates.shape[1] != 2 or len(np.unique(truth)) != 3:
                raise ValueError("This topic visualization requires three communities and two display coordinates.")
            with (folder / "coordinates.csv").open("w", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["node", "x_pc1_lsym", "y_pc2_lsym", "truth", "L_aligned",
                                 "Lsym_aligned", "L_error", "Lsym_error"])
                for i, (x, y) in enumerate(coordinates):
                    writer.writerow([i, x, y, truth[i], predictions[0][i], predictions[1][i],
                                     int(predictions[0][i] != truth[i]), int(predictions[1][i] != truth[i])])
            low, high = coordinates.min(axis=0), coordinates.max(axis=0)
            pad = np.maximum((high - low) * 0.08, 1e-5)
            for col, (labels, title) in enumerate(zip(
                    [truth, *predictions], ["True communities", r"$L$ prediction", r"$L_{\mathrm{sym}}$ prediction"])):
                ax = axes[col]
                colors = [PALETTE[label] if label >= 0 else "#888888" for label in labels]
                ax.scatter(*coordinates.T, c=colors, s=20, alpha=0.8, linewidths=0)
                wrong = labels != truth
                if col:
                    ax.scatter(*coordinates[wrong].T, facecolors="none", edgecolors="#222222",
                               s=33, linewidths=0.65)
                subtitle = f"{wrong.sum()}/{len(truth)} nodes misclassified" if col else "Reference labels"
                ax.set(title=f"r = {ratio:g}  |  {title}\n{subtitle}", xlabel="PC1 of Lsym features",
                       ylabel="PC2 of Lsym features", xlim=(low[0] - pad[0], high[0] + pad[0]),
                       ylim=(low[1] - pad[1], high[1] + pad[1]))
                ax.set_aspect("equal", adjustable="box")
                ax.grid(alpha=0.15)
                ax.set_axisbelow(True)
            handles = [Line2D([], [], color=color, marker="o", linestyle="none", label=f"Group {i + 1}")
                       for i, color in enumerate(PALETTE)]
            handles.append(Line2D([], [], color="#222222", marker="o", markerfacecolor="none",
                                  linestyle="none", label="Misclassified node"))
            legend_ax = fig.add_subplot(grid[-1, :])
            legend_ax.axis("off")
            legend_ax.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
            fig.suptitle(f"Community contrast r = {ratio:g} | repeat 0 | n = {len(truth)}, k = 3\n"
                         "Same node positions; colors show true or matched predicted labels",
                         fontsize=13)
            legend_ax.text(0.5, 0, f"2D PCA display of Lsym features ({pca.explained_variance_ratio_.sum():.1%} variance) | clustering uses all 3 features",
                           ha="center", va="bottom", fontsize=9, transform=legend_ax.transAxes)
            fig.savefig(folder / "embedding.png", dpi=180)
            # Add every observed undirected edge once, without changing the layout
            # or rerunning clustering. Faint lines keep dense graphs readable.
            edge_i, edge_j = np.nonzero(np.triu(adjacency > 0, k=1))
            segments = np.stack((coordinates[edge_i], coordinates[edge_j]), axis=1)
            for ax in axes:
                ax.add_collection(LineCollection(segments, colors="#555555", linewidths=0.35,
                                                  alpha=0.10, zorder=0.5))
            fig.suptitle(f"Observed graph | r = {ratio:g} | repeat 0 | {len(truth)} nodes, {len(edge_i)} edges\n"
                         "Same positions and colors as the scatter plot; gray lines show all observed edges",
                         fontsize=13)
            fig.savefig(folder / "graph.png", dpi=180)
            plt.close(fig)
