"""Explicit Laplacians and spectral embeddings, followed by k-means."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

METHODS = ("unnormalized", "symmetric")


@dataclass
class SpectralResult:
    labels: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    embedding: np.ndarray
    eigengap: float
    isolated_nodes: int
    connected_components: int
    clusters_found: int


def laplacian(adjacency, method):
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1] or len(adjacency) < 2:
        raise ValueError("adjacency must be a square matrix with at least two nodes")
    if not np.isfinite(adjacency).all() or np.any(adjacency < 0):
        raise ValueError("adjacency must be finite and nonnegative")
    if not np.allclose(adjacency, adjacency.T, atol=1e-12, rtol=0):
        raise ValueError("adjacency must be symmetric")
    if np.any(np.diag(adjacency) != 0):
        raise ValueError("self-loops are not supported")
    adjacency = (adjacency + adjacency.T) / 2
    degrees = adjacency.sum(axis=1)
    matrix = np.diag(degrees) - adjacency
    if method == "symmetric":
        inverse_root = np.zeros_like(degrees)
        positive = degrees > 0
        inverse_root[positive] = 1 / np.sqrt(degrees[positive])
        matrix = inverse_root[:, None] * matrix * inverse_root[None, :]
    return matrix, degrees


def spectral_cluster(adjacency, n_clusters, method="symmetric", seed=0, n_init=20):
    """Use the k smallest eigenvectors, including the trivial eigenvector.

    Symmetric mode applies row normalization before k-means (Ng-Jordan-Weiss
    convention); unnormalized mode uses the eigenvectors without row scaling.
    Isolates are retained and scored, with zero rows/columns in either Laplacian.
    """
    matrix, degrees = laplacian(adjacency, method)
    n = len(matrix)
    if (isinstance(n_clusters, bool) or not isinstance(n_clusters, (int, np.integer))
            or not 1 <= n_clusters < n):
        raise ValueError("n_clusters must be an integer in [1, n-1]")
    if isinstance(n_init, bool) or not isinstance(n_init, (int, np.integer)) or n_init < 1:
        raise ValueError("n_init must be a positive integer")
    # Small baseline graphs: a full dense decomposition makes diagnostics explicit.
    with threadpool_limits(limits=1):
        eigenvalues, eigenvectors = eigh(matrix, driver="evd")
        embedding = eigenvectors[:, :n_clusters].copy()
        if method == "symmetric":
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            embedding = np.divide(embedding, norms, out=np.zeros_like(embedding), where=norms > 0)
        labels = KMeans(n_clusters=n_clusters, n_init=int(n_init),
                        random_state=seed, algorithm="lloyd").fit_predict(embedding)
    return SpectralResult(
        labels=labels,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors[:, :n_clusters],
        embedding=embedding,
        eigengap=float(eigenvalues[n_clusters] - eigenvalues[n_clusters - 1]),
        isolated_nodes=int(np.count_nonzero(degrees == 0)),
        connected_components=int(connected_components(adjacency, directed=False, return_labels=False)),
        clusters_found=int(len(np.unique(labels))),
    )
