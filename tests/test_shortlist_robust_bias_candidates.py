import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "shortlist_robust_bias_candidates.py"
)
SPEC = importlib.util.spec_from_file_location(
    "shortlist_robust_bias_candidates", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def baseline():
    return {"bias_mean_pair_target_probability": 0.1}


def row(candidate_id, reduction, retention=1.0):
    return {
        "candidate_id": candidate_id,
        "bias_absolute_gap_reduction": reduction,
        "bias_mean_pair_target_probability": 0.1 * retention,
    }


class ShortlistRobustBiasCandidatesTest(unittest.TestCase):
    def test_requires_every_environment_to_pass(self):
        candidates = [
            {"candidate_id": "robust"},
            {"candidate_id": "fails_lexical"},
            {"candidate_id": "low_retention"},
        ]
        baselines = {
            environment: baseline() for environment in MODULE.ENVIRONMENTS
        }
        rows = {
            "iid": [
                row("robust", 0.1),
                row("fails_lexical", 0.1),
                row("low_retention", 0.1),
            ],
            "lexical_ood": [
                row("robust", 0.1),
                row("fails_lexical", -0.01),
                row("low_retention", 0.1, 0.9),
            ],
            "template_ood": [
                row("robust", 0.1),
                row("fails_lexical", 0.1),
                row("low_retention", 0.1),
            ],
        }
        shortlisted = MODULE.robust_bias_shortlist(
            candidates, baselines, rows, 0.95, 0.0
        )
        self.assertEqual(
            [candidate["candidate_id"] for candidate in shortlisted],
            ["robust"],
        )
        self.assertIn("bias_validation", shortlisted[0])


if __name__ == "__main__":
    unittest.main()
