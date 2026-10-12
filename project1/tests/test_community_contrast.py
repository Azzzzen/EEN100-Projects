"""Check the topic's controlled design and reported uncertainty."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.stats import binomtest

from spectral_project.sbm import expected_average_degree

path = Path(__file__).resolve().parents[1] / "topics/community_contrast/run.py"
spec = importlib.util.spec_from_file_location("community_contrast", path)
topic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(topic)


class ContrastTests(unittest.TestCase):
    def test_controlled_design(self):
        seeds = []
        for ratio, config in topic.settings():
            probabilities = np.asarray(config["block_probabilities"])
            self.assertEqual(config["sizes"], [100, 50, 30])
            self.assertEqual(config["repeats"], 50)
            self.assertAlmostEqual(expected_average_degree(config["sizes"], probabilities), 12)
            per_group = probabilities @ config["sizes"] - np.diag(probabilities)
            self.assertAlmostEqual(np.average(per_group, weights=config['sizes']), 12)
            if ratio > 1:
                self.assertGreater(per_group[0], per_group[-1])
            if ratio == 5:
                self.assertEqual(config['seed'], 20260928)
            self.assertAlmostEqual(probabilities[0, 0] / probabilities[0, 1], ratio)
            seeds.append(config["seed"])
        self.assertEqual(len(set(seeds)), len(topic.RATIOS))

    def test_wilson_intervals(self):
        for successes in (0, 1, 10, 25, 49, 50):
            reference = binomtest(successes, 50).proportion_ci(method="wilson")
            np.testing.assert_allclose(topic.wilson_interval(successes, 50),
                                       (reference.low, reference.high), atol=1e-14)
