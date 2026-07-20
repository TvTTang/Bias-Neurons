import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "subset_gap_attributions.py"
)
SPEC = importlib.util.spec_from_file_location("subset_gap_attributions", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def manifest():
    return {
        "bag_partitions": {"train": [2, 0], "val": [1], "test": [3]},
        "template_partitions": {
            "train": [2, 0],
            "val": [1],
            "test": [3],
        },
        "splits": {
            "train": {
                "bag_partition": "train",
                "template_partition": "train",
            }
        },
    }


class SubsetGapAttributionsTest(unittest.TestCase):
    def test_subsets_both_axes_and_preserves_manifest_template_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "gap.jsonl"
            split_manifest = root / "manifest.json"
            output = root / "train.jsonl"
            with source.open("w", encoding="utf-8") as handle:
                for bag in range(4):
                    handle.write(
                        json.dumps(
                            [[f"bag{bag}-template{template}"] for template in range(4)]
                        )
                        + "\n"
                    )
            split_manifest.write_text(
                json.dumps(manifest()), encoding="utf-8"
            )

            summary = MODULE.subset_gap(
                source, split_manifest, "train", output
            )

            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                rows,
                [
                    [["bag0-template2"], ["bag0-template0"]],
                    [["bag2-template2"], ["bag2-template0"]],
                ],
            )
            self.assertEqual(summary["selected_bag_indices_source_order"], [0, 2])
            self.assertEqual(summary["output_num_examples"], 4)

    def test_non_exhaustive_partitions_are_rejected(self):
        bad = manifest()
        bad["bag_partitions"]["test"] = [4]
        with self.assertRaisesRegex(ValueError, "contiguous"):
            MODULE.partition_size(bad["bag_partitions"], "Bag")

    def test_refuses_to_overwrite_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "gap.jsonl"
            split_manifest = root / "manifest.json"
            output = root / "train.jsonl"
            source.write_text("[]\n", encoding="utf-8")
            split_manifest.write_text(json.dumps(manifest()), encoding="utf-8")
            output.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                MODULE.subset_gap(source, split_manifest, "train", output)


if __name__ == "__main__":
    unittest.main()
