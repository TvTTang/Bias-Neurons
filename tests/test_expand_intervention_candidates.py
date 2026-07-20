import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "expand_intervention_candidates.py"
)
SPEC = importlib.util.spec_from_file_location(
    "expand_intervention_candidates", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ExpandInterventionCandidatesTest(unittest.TestCase):
    def test_unique_sets_choose_stable_low_penalty_representative(self):
        candidates = [
            {
                "candidate_id": "low",
                "neurons": [[1, 2]],
                "stability_min": 0.25,
                "semantic_weight": 0.25,
            },
            {
                "candidate_id": "stable",
                "neurons": [[1, 2]],
                "stability_min": 0.75,
                "semantic_weight": 0.5,
            },
            {
                "candidate_id": "stable-low-penalty",
                "neurons": [[1, 2]],
                "stability_min": 0.75,
                "semantic_weight": 0.25,
            },
        ]
        unique = MODULE.unique_source_candidates(candidates)
        self.assertEqual(unique[0]["candidate_id"], "stable-low-penalty")

    def test_expansion_preserves_both_intervention_directions(self):
        candidates = [
            {
                "candidate_id": "source",
                "neurons": [[1, 2]],
                "stability_min": 0.75,
                "semantic_weight": 0.25,
            }
        ]
        expanded = MODULE.expand_candidates(candidates, [0.5, 1.5])
        self.assertEqual(
            [item["candidate_id"] for item in expanded],
            ["source_scale-0p5", "source_scale-1p5"],
        )
        self.assertEqual(
            [item["intervention_scale"] for item in expanded],
            [0.5, 1.5],
        )

    def test_baseline_scale_is_rejected(self):
        candidates = [
            {
                "candidate_id": "source",
                "neurons": [[1, 2]],
                "stability_min": 0.75,
                "semantic_weight": 0.25,
            }
        ]
        with self.assertRaisesRegex(ValueError, "baseline"):
            MODULE.expand_candidates(candidates, [1.0])


if __name__ == "__main__":
    unittest.main()
