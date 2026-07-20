import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "freeze_validation_candidate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "freeze_validation_candidate", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


THRESHOLDS = {
    "min_target_probability_retention": 0.95,
    "min_top1_agreement": 0.995,
    "max_mean_kl": 1e-4,
    "max_nll_increase": 0.005,
    "min_bias_reduction": 0.0,
}


def row(candidate_id, reduction, target_probability=0.1):
    return {
        "candidate_id": candidate_id,
        "num_neurons": 2.0,
        "bias_mean_pair_target_probability": target_probability,
        "bias_absolute_gap_reduction": reduction,
        "semantic_baseline_top1_agreement": 1.0,
        "semantic_mean_kl_from_baseline": 1e-5,
        "semantic_nll_increase": 0.001,
    }


class FreezeValidationCandidateTest(unittest.TestCase):
    def setUp(self):
        self.baseline = {"bias_mean_pair_target_probability": 0.1}
        self.candidate_sets = {
            "lower-stability": {
                "stability_min": 0.5,
                "semantic_weight": 0.25,
                "neurons": [[1, 2], [3, 4]],
            },
            "selected": {
                "stability_min": 0.75,
                "semantic_weight": 0.25,
                "neurons": [[1, 2], [3, 4]],
            },
            "unsafe": {
                "stability_min": 0.75,
                "semantic_weight": 0.5,
                "neurons": [[5, 6], [7, 8]],
            },
        }

    def test_constraints_and_tie_break_are_applied(self):
        rows = [
            row("lower-stability", 0.2),
            row("selected", 0.2),
            row("unsafe", 0.3, target_probability=0.09),
        ]
        assessed = MODULE.assess_candidates(
            rows, self.baseline, self.candidate_sets, THRESHOLDS
        )
        frozen = MODULE.freeze(
            "example",
            self.baseline,
            assessed,
            self.candidate_sets,
            THRESHOLDS,
        )
        self.assertEqual(frozen["status"], "intervention")
        self.assertEqual(frozen["selected_candidate_id"], "selected")
        self.assertEqual(frozen["num_feasible_candidates"], 2)

    def test_no_positive_candidate_freezes_no_intervention(self):
        candidate_sets = {
            "worse": {
                "stability_min": 0.75,
                "semantic_weight": 0.25,
                "neurons": [[1, 2]],
            }
        }
        rows = [row("worse", -0.1)]
        assessed = MODULE.assess_candidates(
            rows, self.baseline, candidate_sets, THRESHOLDS
        )
        frozen = MODULE.freeze(
            "example",
            self.baseline,
            assessed,
            candidate_sets,
            THRESHOLDS,
        )
        self.assertEqual(frozen["status"], "no_intervention")
        self.assertIsNone(frozen["selected_candidate_id"])
        self.assertEqual(frozen["selected_neurons"], [])


if __name__ == "__main__":
    unittest.main()
