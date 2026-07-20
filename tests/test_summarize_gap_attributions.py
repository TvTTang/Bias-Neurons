import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "summarize_gap_attributions.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_gap_attributions", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SummarizeGapAttributionsTest(unittest.TestCase):
    def test_metric_aliases_are_supported(self):
        self.assertEqual(
            MODULE.resolve_metric({"ig2_gold_gap": []}, "auto"), "ig2_gold_gap"
        )
        self.assertEqual(
            MODULE.resolve_metric({"ig_gold_gap": []}, "auto"), "ig_gold_gap"
        )

    def test_relation_name(self):
        path = Path(
            "Modifier-gender-N-filtered-gap-rm-base-male-female.rlt.jsonl"
        )
        self.assertEqual(MODULE.relation_name(path), "male-female")


if __name__ == "__main__":
    unittest.main()
