import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "build_semantic_calibration.py"
)
SPEC = importlib.util.spec_from_file_location("build_semantic_calibration", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BuildSemanticCalibrationTest(unittest.TestCase):
    def test_wikitext_markup_is_normalized(self):
        self.assertEqual(
            MODULE.normalize_text("A role @-@ playing game @,@ released ."),
            "A role-playing game, released .",
        )

    def test_headings_are_excluded_and_sentences_are_split(self):
        lines = [" = Heading = \n", "First sentence. Second sentence!\n"]
        self.assertEqual(
            list(MODULE.sentence_candidates(lines)),
            ["First sentence.", "Second sentence!"],
        )

    def test_filter_pattern_uses_term_boundaries(self):
        pattern = MODULE.compile_filter_pattern(["male", "working-class"])
        self.assertIsNotNone(pattern.search("A male participant arrived."))
        self.assertIsNotNone(pattern.search("A working-class family arrived."))
        self.assertIsNone(pattern.search("The word femalevolent is synthetic."))

    def test_only_complete_wordpieces_are_masked(self):
        tokens = ["play", "##ing", "works", ".", "[UNK]"]
        self.assertEqual(MODULE.valid_mask_positions(tokens), [2])

    def test_hash_selection_is_deterministic(self):
        first = MODULE.hash_score(42, "train", "A sentence.")
        second = MODULE.hash_score(42, "train", "A sentence.")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
