import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "experiments" / "extract_bias_neurons.py"
)
SPEC = importlib.util.spec_from_file_location("extract_bias_neurons", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ExtractBiasNeuronsTest(unittest.TestCase):
    def test_auto_metric_and_original_frequency_rule(self):
        bags = []
        for _ in range(4):
            examples = []
            for _ in range(5):
                examples.append(
                    [
                        {"tokens": ["[MASK]"]},
                        {
                            "ig2_gold_gap": [
                                [10, 7, 1.0],
                                [11, 9, 0.3],
                                [3, 2, 0.1],
                            ]
                        },
                    ]
                )
            bags.append(examples)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.rlt.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for bag in bags:
                    handle.write(json.dumps(bag) + "\n")

            counts, sizes, metric = MODULE.read_bag_counts(path, "auto", 0.2)
            ratio, average, selected = MODULE.choose_bag_neurons(
                counts, sizes, 0.7, 3, 6, 2.0, 5.0
            )

        self.assertEqual(metric, "ig2_gold_gap")
        self.assertEqual(sizes, [5, 5, 5, 5])
        self.assertEqual(ratio, 0.7)
        self.assertEqual(average, 2.0)
        self.assertEqual(selected[0], [(10, 7), (11, 9)])

    def test_released_legacy_metric_name_is_supported(self):
        result = {"ig_gold_gap": [[1, 2, 0.5]]}
        self.assertEqual(MODULE.resolve_metric(result, "auto"), "ig_gold_gap")

    def test_non_finite_values_are_rejected(self):
        with self.assertRaises(ValueError):
            list(MODULE.filtered_positions([[1, 2, float("nan")]], 0.2))


if __name__ == "__main__":
    unittest.main()
