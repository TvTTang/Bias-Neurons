import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "evaluate_neuron_candidates.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_neuron_candidates", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class EvaluateNeuronCandidatesTest(unittest.TestCase):
    def test_bias_summary(self):
        summary = MODULE.summarize_bias(
            signed_sum=0.2,
            absolute_sum=0.4,
            square_sum=0.1,
            target_probability_sum=0.6,
            count=2,
        )
        self.assertAlmostEqual(summary["bias_mean_signed_gap"], 0.1)
        self.assertAlmostEqual(summary["bias_mean_absolute_gap"], 0.2)
        self.assertAlmostEqual(summary["bias_rms_gap"], (0.05) ** 0.5)

    def test_candidate_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.json"
            path.write_text(
                json.dumps(
                    [
                        {"candidate_id": "a", "neurons": [[1, 2]]},
                        {"candidate_id": "b", "neurons": [[3, 4]]},
                    ]
                ),
                encoding="utf-8",
            )
            selected = MODULE.load_candidates(path, ["b"])
            self.assertEqual(selected[0]["candidate_id"], "b")
            self.assertEqual(selected[0]["neurons"], [(3, 4)])

    def test_duplicate_neuron_sets_are_evaluated_once(self):
        candidates = [
            {"candidate_id": "a", "neurons": [(1, 2), (3, 4)]},
            {"candidate_id": "b", "neurons": [(1, 2), (3, 4)]},
            {"candidate_id": "c", "neurons": [(5, 6)]},
        ]
        unique, mapping = MODULE.deduplicate_candidates(candidates)
        self.assertEqual([item["candidate_id"] for item in unique], ["a", "c"])
        self.assertEqual(mapping, {"a": "a", "b": "a", "c": "c"})

    def test_paired_prompt_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dimension = root / "gender"
            dimension.mkdir()
            (dimension / "male_N_data.json").write_text(
                json.dumps([[["A [MASK]", "male", "r"]]]),
                encoding="utf-8",
            )
            (dimension / "female_N_data.json").write_text(
                json.dumps([[["B [MASK]", "female", "r"]]]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "prompts differ"):
                MODULE.load_bias_pairs(
                    root, "gender", "male", "female", "N"
                )


if __name__ == "__main__":
    unittest.main()
