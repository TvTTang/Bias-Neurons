import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "freeze_robust_validation_candidate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "freeze_robust_validation_candidate", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


THRESHOLDS = {
    "min_target_probability_retention": 0.95,
    "min_top1_agreement": 0.995,
    "max_mean_kl": 1e-4,
    "max_nll_increase": 0.005,
    "min_environment_bias_reduction": 0.0,
}


def baseline():
    return {
        "bias_mean_pair_target_probability": 0.1,
        "bias_mean_absolute_gap": 0.2,
    }


def row(candidate_id, reduction, retention=1.0):
    return {
        "candidate_id": candidate_id,
        "num_neurons": 2.0,
        "num_bias_examples": 20.0,
        "bias_mean_absolute_gap": 0.2 * (1 - reduction),
        "bias_mean_pair_target_probability": 0.1 * retention,
        "bias_absolute_gap_reduction": reduction,
        "num_semantic_examples": 100.0,
        "semantic_baseline_top1_agreement": 1.0,
        "semantic_mean_kl_from_baseline": 1e-5,
        "semantic_nll_increase": 0.001,
    }


class FreezeRobustValidationCandidateTest(unittest.TestCase):
    def setUp(self):
        self.baselines = {
            environment: baseline() for environment in MODULE.ENVIRONMENTS
        }
        self.candidates = {
            "robust": {
                "neurons": [[1, 2], [3, 4]],
                "intervention_scale": 0.5,
                "stability_min": 0.75,
                "semantic_weight": 0.25,
            },
            "high-average": {
                "neurons": [[5, 6], [7, 8]],
                "intervention_scale": 1.5,
                "stability_min": 0.75,
                "semantic_weight": 0.25,
            },
        }

    def test_max_min_prefers_robust_candidate(self):
        reductions = {
            "iid": {"robust": 0.1, "high-average": 0.3},
            "lexical_ood": {"robust": 0.1, "high-average": 0.01},
            "template_ood": {"robust": 0.1, "high-average": 0.3},
        }
        rows = {
            environment: [
                row(candidate_id, reductions[environment][candidate_id])
                for candidate_id in self.candidates
            ]
            for environment in MODULE.ENVIRONMENTS
        }
        assessed = MODULE.assess_robust_candidates(
            self.baselines, rows, self.candidates, THRESHOLDS
        )
        frozen = MODULE.freeze_robust(
            "example",
            self.baselines,
            assessed,
            self.candidates,
            THRESHOLDS,
        )
        self.assertEqual(frozen["selected_candidate_id"], "robust")
        self.assertEqual(frozen["selected_intervention_scale"], 0.5)

    def test_one_failed_environment_rejects_candidate(self):
        rows = {
            "iid": [row("robust", 0.1)],
            "lexical_ood": [row("robust", -0.01)],
            "template_ood": [row("robust", 0.1)],
        }
        candidates = {"robust": self.candidates["robust"]}
        assessed = MODULE.assess_robust_candidates(
            self.baselines, rows, candidates, THRESHOLDS
        )
        frozen = MODULE.freeze_robust(
            "example",
            self.baselines,
            assessed,
            candidates,
            THRESHOLDS,
        )
        self.assertEqual(frozen["status"], "no_intervention")

    def test_extra_environment_rows_are_ignored_after_shortlisting(self):
        candidates = {"robust": self.candidates["robust"]}
        rows = {
            environment: [
                row("robust", 0.1),
                row("not_shortlisted", -0.1),
            ]
            for environment in MODULE.ENVIRONMENTS
        }
        assessed = MODULE.assess_robust_candidates(
            self.baselines, rows, candidates, THRESHOLDS
        )
        self.assertEqual(
            [candidate["candidate_id"] for candidate in assessed],
            ["robust"],
        )


if __name__ == "__main__":
    unittest.main()
