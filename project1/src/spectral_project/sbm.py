"""Undirected, unweighted stochastic block models without self-loops."""

import numpy as np


def validate_sbm(sizes, block_probabilities):
    sizes = np.asarray(sizes)
    if (sizes.ndim != 1 or sizes.size == 0
            or not np.issubdtype(sizes.dtype, np.integer) or np.any(sizes <= 0)):
        raise ValueError("sizes must be a nonempty list of positive integers")
    probabilities = np.asarray(block_probabilities, dtype=float)
    k = len(sizes)
    if probabilities.shape != (k, k):
        raise ValueError("block_probabilities must have shape (len(sizes), len(sizes))")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("block probabilities must be finite and in [0, 1]")
    if not np.allclose(probabilities, probabilities.T, atol=1e-12, rtol=0):
        raise ValueError("block_probabilities must be symmetric")
    return sizes.astype(int), (probabilities + probabilities.T) / 2


def expected_average_degree(sizes, block_probabilities):
    """Account for exclusion of each node's self-loop."""
    sizes, probabilities = validate_sbm(sizes, block_probabilities)
    degrees = probabilities @ sizes - np.diag(probabilities)
    return float(np.dot(sizes, degrees) / sizes.sum())


def sample_sbm(sizes, block_probabilities, seed):
    """Return (W, truth); sample each unordered edge once, then shuffle nodes.

    Shuffling prevents downstream code from relying on contiguous true blocks.
    Ground-truth labels are never used by the clustering function.
    """
    sizes, probabilities = validate_sbm(sizes, block_probabilities)
    rng = np.random.default_rng(seed)
    truth = np.repeat(np.arange(len(sizes)), sizes)
    n = len(truth)
    rows, cols = np.triu_indices(n, k=1)
    edges = rng.random(len(rows)) < probabilities[truth[rows], truth[cols]]
    adjacency = np.zeros((n, n), dtype=float)
    adjacency[rows, cols] = edges
    adjacency[cols, rows] = edges
    order = rng.permutation(n)
    return adjacency[np.ix_(order, order)], truth[order]
