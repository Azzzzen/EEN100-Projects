"""Scientific correctness checks using Python's built-in unittest runner."""

import unittest
import importlib.util
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from spectral_project.experiment import make_config, run_experiment, validate_config
from spectral_project.metrics import recovery_metrics
from spectral_project.sbm import expected_average_degree, sample_sbm
from spectral_project.spectral import laplacian, spectral_cluster


class SBMTests(unittest.TestCase):
    def test_disconnected_cliques_and_seed(self):
        adjacency, truth = sample_sbm([5, 7], [[1, 0], [0, 1]], seed=42)
        expected = (truth[:, None] == truth[None, :]).astype(float)
        np.fill_diagonal(expected, 0)
        assert_array_equal(adjacency, expected)
        other, other_truth = sample_sbm([5, 7], [[1, 0], [0, 1]], seed=42)
        assert_array_equal(adjacency, other)
        assert_array_equal(truth, other_truth)
        self.assertAlmostEqual(expected_average_degree([5, 7], [[1, 0], [0, 1]]), adjacency.sum() / 12)

    def test_invalid_models(self):
        for sizes, probabilities in [([3.5, 4], [[1, 0], [0, 1]]),
                                     ([3, 4], [[0.5, 0.2], [0.1, 0.5]]),
                                     ([3], [[1.1]]), ([3], [[float("nan")]])]:
            with self.subTest(sizes=sizes, probabilities=probabilities), self.assertRaises(ValueError):
                sample_sbm(sizes, probabilities, 0)


class SpectralTests(unittest.TestCase):
    def test_laplacian_identities(self):
        adjacency = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]], dtype=float)
        matrix, degrees = laplacian(adjacency, "unnormalized")
        vector = np.array([1, 3, -2, 4])
        self.assertAlmostEqual(float(vector @ matrix @ vector),
                               0.5 * np.sum(adjacency * (vector[:, None] - vector) ** 2))
        assert_allclose(matrix @ np.ones(4), 0, atol=1e-12)
        normalized, _ = laplacian(adjacency, "symmetric")
        assert_allclose(normalized @ np.sqrt(degrees), 0, atol=1e-12)
        values = np.linalg.eigvalsh(normalized)
        self.assertGreaterEqual(values.min(), -1e-12)
        self.assertLessEqual(values.max(), 2 + 1e-12)

    def test_recover_disconnected_cliques(self):
        adjacency, truth = sample_sbm([6, 8, 10], np.eye(3), seed=4)
        for method in ("unnormalized", "symmetric"):
            result = spectral_cluster(adjacency, 3, method, seed=3)
            self.assertEqual(recovery_metrics(truth, result.labels)["exact_recovery"], 1)
            assert_allclose(result.eigenvalues[:3], 0, atol=1e-10)
            self.assertGreater(result.eigengap, 0.5)
            self.assertEqual(result.connected_components, 3)
            if method == "symmetric":
                assert_allclose(np.linalg.norm(result.embedding, axis=1), 1)

    def test_isolates_retained_and_finite(self):
        adjacency = np.zeros((8, 8))
        adjacency[:4, :4] = 1 - np.eye(4)
        for method in ("unnormalized", "symmetric"):
            result = spectral_cluster(adjacency, 2, method, seed=0)
            self.assertEqual(result.isolated_nodes, 4)
            self.assertEqual(result.connected_components, 5)
            self.assertEqual(len(result.labels), 8)
            self.assertTrue(np.isfinite(result.embedding).all())
            self.assertTrue(np.isfinite(result.eigenvalues).all())

    def test_invalid_graph(self):
        for adjacency in [np.eye(3), np.array([[0, 1], [0, 0]]), np.zeros((2, 3))]:
            with self.assertRaises(ValueError):
                laplacian(adjacency, "symmetric")


class MetricsTests(unittest.TestCase):
    def test_permutation_invariance(self):
        scores = recovery_metrics([0, 0, 1, 1, 2, 2], [8, 8, 3, 3, 5, 5])
        self.assertEqual(scores["misclassification_rate"], 0)
        self.assertEqual(scores["exact_recovery"], 1)
        self.assertEqual(scores["ari"], 1)

    def test_collapsed_and_split_clusters(self):
        scores = recovery_metrics([0, 0, 1, 1], [7, 7, 7, 7])
        self.assertEqual(scores["misclassification_rate"], 0.5)
        self.assertEqual(scores["exact_recovery"], 0)
        scores = recovery_metrics([0, 0, 1, 1], [1, 2, 3, 4])
        self.assertEqual(scores["misclassification_rate"], 0.5)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.config = {"name": "check", "sizes": [12, 12],
                       "block_probabilities": [[0.7, 0.02], [0.02, 0.7]],
                       "methods": ["unnormalized", "symmetric"], "seed": 91,
                       "repeats": 3, "n_init": 5}

    def test_paired_graphs_reproducibility_and_method_order(self):
        rows, summary, example = run_experiment(self.config)
        again, _, second_example = run_experiment(self.config)
        self.assertEqual(rows, again)
        assert_array_equal(example["adjacency"], second_example["adjacency"])
        self.assertEqual(len(summary), 2)
        for index in range(0, len(rows), 2):
            self.assertEqual(rows[index]["graph_seed"], rows[index + 1]["graph_seed"])
            self.assertEqual(rows[index]["average_degree"], rows[index + 1]["average_degree"])
        self.assertEqual(len({row["graph_seed"] for row in rows}), 3)
        reversed_rows, _, _ = run_experiment({**self.config, "methods": self.config["methods"][::-1]})
        key = lambda row: (row["repeat"], row["method"])
        self.assertEqual(sorted(rows, key=key), sorted(reversed_rows, key=key))

    def test_invalid_config_and_single_repeat(self):
        for change in ({"repeats": 0}, {"seed": -1}, {"methods": ["typo"]}, {"extra": True}):
            with self.assertRaises(ValueError):
                validate_config({**self.config, **change})
        _, summary, _ = run_experiment({**self.config, "repeats": 1})
        self.assertIsNone(summary[0]["misclassification_rate_sd"])

    def test_make_config_matches_handwritten(self):
        config = make_config("check", [12, 12], 0.7, 0.02, repeats=3, seed=91, n_init=5)
        self.assertEqual(config, self.config)
        rows, summary, _ = run_experiment(config)
        self.assertEqual({row["name"] for row in rows + summary}, {"check"})
        with self.assertRaises(ValueError):
            make_config("bad", [5, 5], 1.2, 0.1)

    def test_make_config_rejects_noninteger_sizes(self):
        for sizes in ([3.5, 4], [True, True], ["3", "4"]):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                make_config("bad", sizes, 0.5, 0.1)

    def test_template_matches_requested_average_degree(self):
        path = Path(__file__).resolve().parents[1] / "topics/_template/run.py"
        spec = importlib.util.spec_from_file_location("topic_template", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for sizes in ([100, 100, 100], [150, 100, 50]):
            module.SIZES = sizes
            for target, config in module.settings():
                actual = expected_average_degree(config["sizes"], config["block_probabilities"])
                self.assertAlmostEqual(actual, target)


if __name__ == "__main__":
    unittest.main()
