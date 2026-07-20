import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "select_semantic_safe_neurons.py"
)
SPEC = importlib.util.spec_from_file_location(
    "select_semantic_safe_neurons", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SelectSemanticSafeNeuronsTest(unittest.TestCase):
    def test_zero_ties_map_to_zero_percentile(self):
        values = {(0, 0): 0.0, (0, 1): 0.0, (0, 2): 1.0}
        normalized = MODULE.zero_origin_percentiles(values)
        self.assertEqual(normalized[(0, 0)], 0.0)
        self.assertEqual(normalized[(0, 1)], 0.0)
        self.assertEqual(normalized[(0, 2)], 1.0)

    def test_semantic_penalty_changes_top_candidate(self):
        bias = {
            (0, 0): 3.0,
            (0, 1): 2.0,
            (0, 2): 1.0,
        }
        semantic = {
            (0, 0): 3.0,
            (0, 1): 0.0,
            (0, 2): 1.0,
        }
        rows = MODULE.build_rows(
            [("bias", bias)], semantic, aggregation="mean"
        )
        candidates = MODULE.candidate_grid(
            rows, semantic_weights=[0.0, 2.0], neuron_counts=[1]
        )
        self.assertEqual(candidates[0]["neurons"], [[0, 0]])
        self.assertEqual(candidates[1]["neurons"], [[0, 1]])

    def test_index_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            MODULE.validate_same_neurons(
                [
                    ("a", {(0, 0): 1.0}),
                    ("b", {(0, 1): 1.0}),
                ]
            )

    def test_pareto_frontier_excludes_dominated_neuron(self):
        rows = [
            {
                "layer": 0,
                "neuron": 0,
                "bias_utility": 1.0,
                "semantic_cost": 0.2,
            },
            {
                "layer": 0,
                "neuron": 1,
                "bias_utility": 0.8,
                "semantic_cost": 0.3,
            },
            {
                "layer": 0,
                "neuron": 2,
                "bias_utility": 0.7,
                "semantic_cost": 0.1,
            },
        ]
        frontier = MODULE.pareto_frontier(rows)
        self.assertIn((0, 0), frontier)
        self.assertIn((0, 2), frontier)
        self.assertNotIn((0, 1), frontier)

    def test_equivalent_neurons_are_both_on_frontier(self):
        rows = [
            {
                "layer": 0,
                "neuron": neuron,
                "bias_utility": 1.0,
                "semantic_cost": 0.2,
            }
            for neuron in (0, 1)
        ]
        self.assertEqual(
            MODULE.pareto_frontier(rows), {(0, 0), (0, 1)}
        )


if __name__ == "__main__":
    unittest.main()
