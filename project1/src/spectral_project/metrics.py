"""Recovery scores invariant to arbitrary cluster label numbering."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


def recovery_metrics(truth, predicted):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    if truth.ndim != 1 or truth.size == 0 or predicted.shape != truth.shape:
        raise ValueError("truth and predicted must be nonempty 1D arrays of equal length")
    _, true_indices = np.unique(truth, return_inverse=True)
    _, pred_indices = np.unique(predicted, return_inverse=True)
    counts = np.zeros((true_indices.max() + 1, pred_indices.max() + 1), dtype=int)
    np.add.at(counts, (true_indices, pred_indices), 1)
    rows, cols = linear_sum_assignment(-counts)
    correct = int(counts[rows, cols].sum())
    errors = len(truth) - correct
    return {
        "misclassified": errors,
        "misclassification_rate": errors / len(truth),
        "exact_recovery": int(errors == 0),
        "ari": float(adjusted_rand_score(truth, predicted)),
    }
