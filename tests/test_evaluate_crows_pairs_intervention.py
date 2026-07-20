import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "evaluate_crows_pairs_intervention.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_crows_pairs_intervention", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class EvaluateCrowsPairsInterventionTest(unittest.TestCase):
    def test_common_positions_match_equal_spans(self):
        first, second = MODULE.official_mask_positions(
            [101, 10, 20, 30, 102],
            [101, 10, 99, 30, 102],
        )
        self.assertEqual(first, [1, 3])
        self.assertEqual(second, [1, 3])

    def test_mcnemar_exact_is_symmetric(self):
        self.assertEqual(
            MODULE.mcnemar_exact_pvalue(8, 2),
            MODULE.mcnemar_exact_pvalue(2, 8),
        )
        self.assertEqual(MODULE.mcnemar_exact_pvalue(0, 0), 1.0)

    def test_summary_uses_paired_direction(self):
        rows = [
            {
                "baseline_biased": 1,
                "intervention_biased": 0,
                "baseline_stereotype_margin": 1.0,
                "intervention_stereotype_margin": -1.0,
            },
            {
                "baseline_biased": 1,
                "intervention_biased": 1,
                "baseline_stereotype_margin": 0.5,
                "intervention_stereotype_margin": 0.25,
            },
        ]
        summary = MODULE.summarize_results(rows, bootstrap_rounds=20, seed=42)
        self.assertEqual(summary["baseline_crows_score"], 1.0)
        self.assertEqual(summary["intervention_crows_score"], 0.5)
        self.assertEqual(summary["crows_score_reduction"], 0.5)
        self.assertEqual(
            summary["baseline_biased_to_intervention_unbiased"], 1
        )


if __name__ == "__main__":
    unittest.main()
