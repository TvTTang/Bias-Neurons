import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "experiments" / "create_paired_splits.py"
)
SPEC = importlib.util.spec_from_file_location("create_paired_splits", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CreatePairedSplitsTest(unittest.TestCase):
    def test_counts_are_exhaustive_and_deterministic(self):
        self.assertEqual(MODULE.allocate_counts(100, (0.6, 0.2, 0.2)), (60, 20, 20))
        self.assertEqual(
            MODULE.allocate_counts(17, (10 / 17, 3 / 17, 4 / 17)), (10, 3, 4)
        )
        first = MODULE.shuffled_partitions(100, (0.6, 0.2, 0.2), 42)
        second = MODULE.shuffled_partitions(100, (0.6, 0.2, 0.2), 42)
        self.assertEqual(first, second)
        flattened = [index for values in first.values() for index in values]
        self.assertEqual(len(flattened), 100)
        self.assertEqual(len(set(flattened)), 100)

    def test_pair_validation_and_subsetting(self):
        group1 = [
            [["prompt-a", "g1", "r1"], ["prompt-b", "g1", "r1"]],
            [["prompt-c", "g1", "r1"], ["prompt-d", "g1", "r1"]],
        ]
        group2 = [
            [["prompt-a", "g2", "r2"], ["prompt-b", "g2", "r2"]],
            [["prompt-c", "g2", "r2"], ["prompt-d", "g2", "r2"]],
        ]
        self.assertEqual(MODULE.validate_pair(group1, group2), 2)
        self.assertEqual(
            MODULE.subset(group1, [1], [0]), [[["prompt-c", "g1", "r1"]]]
        )

    def test_prompt_mismatch_is_rejected(self):
        group1 = [[["prompt-a", "g1", "r1"]]]
        group2 = [[["different", "g2", "r2"]]]
        with self.assertRaises(ValueError):
            MODULE.validate_pair(group1, group2)


if __name__ == "__main__":
    unittest.main()
