import importlib.util
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "select_stable_semantic_safe_neurons.py"
)
SPEC = importlib.util.spec_from_file_location(
    "select_stable_semantic_safe_neurons", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SelectStableSemanticSafeNeuronsTest(unittest.TestCase):
    @unittest.skipIf(np is None, "NumPy is not installed in the local test runtime.")
    def test_direction_strength_penalizes_sign_flips(self):
        means = np.array(
            [
                [1.0, 1.0, 0.0],
                [2.0, -1.0, 0.0],
                [3.0, 1.0, 0.0],
            ]
        )
        strength = MODULE.direction_strength(means, np)
        self.assertEqual(strength[0], 1.0)
        self.assertAlmostEqual(strength[1], 1 / 3)
        self.assertEqual(strength[2], 0.0)

    @unittest.skipIf(np is None, "NumPy is not installed in the local test runtime.")
    def test_bootstrap_frequency_is_deterministic(self):
        means = np.array(
            [
                [3.0, 0.1, 0.0],
                [2.0, 0.2, 0.0],
                [4.0, 0.1, 0.0],
            ]
        )
        first = MODULE.bootstrap_top_frequency(means, 20, 1, 42, np)
        second = MODULE.bootstrap_top_frequency(means, 20, 1, 42, np)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first[0], 1.0)

    def test_candidate_grid_applies_stability_gate(self):
        rows = [
            {
                "layer": 0,
                "neuron": 0,
                "dual_axis_stability": 0.8,
                "stable_bias_utility": 0.9,
                "semantic_cost": 0.8,
            },
            {
                "layer": 0,
                "neuron": 1,
                "dual_axis_stability": 0.6,
                "stable_bias_utility": 0.7,
                "semantic_cost": 0.0,
            },
        ]
        candidates = MODULE.candidate_grid(
            rows,
            stability_thresholds=[0.75],
            semantic_weights=[2.0],
            neuron_counts=[1],
        )
        self.assertEqual(candidates[0]["neurons"], [[0, 0]])

    @unittest.skipIf(np is None, "NumPy is not installed in the local test runtime.")
    def test_zero_ties_have_zero_percentile(self):
        values = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_equal(
            MODULE.zero_origin_percentiles(values, np),
            np.array([0.0, 0.0, 1.0]),
        )


if __name__ == "__main__":
    unittest.main()
