import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "prepare_cased_targets.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_cased_targets", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    unk_token_id = 100
    vocabulary = {"Islam": 6489, "Christianity": 7522}

    def tokenize(self, target):
        return [target] if target in self.vocabulary else ["[UNK]"]

    def convert_tokens_to_ids(self, target):
        return self.vocabulary.get(target, self.unk_token_id)


class PrepareCasedTargetsTest(unittest.TestCase):
    def test_only_gold_target_is_changed(self):
        source = [
            [
                ["A [MASK].", "islam", "relation-islam"],
                ["B [MASK].", "islam", "relation-islam"],
            ]
        ]
        corrected = MODULE.replace_targets(source, "islam", "Islam")
        self.assertEqual(corrected[0][0], ["A [MASK].", "Islam", "relation-islam"])
        self.assertEqual(source[0][0][1], "islam")

    def test_unexpected_source_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unexpected target"):
            MODULE.replace_targets(
                [[["A [MASK].", "other", "relation"]]], "islam", "Islam"
            )

    def test_corrected_targets_are_distinct_single_tokens(self):
        tokenizer = FakeTokenizer()
        self.assertEqual(
            MODULE.validate_single_token(tokenizer, "Islam"), 6489
        )
        self.assertEqual(
            MODULE.validate_single_token(tokenizer, "Christianity"), 7522
        )
        with self.assertRaisesRegex(ValueError, "not preserved"):
            MODULE.validate_single_token(tokenizer, "islam")

    def test_group_mapping_parser(self):
        self.assertEqual(
            MODULE.parse_group_mappings(
                ["islam=Islam", "christianity=Christianity"]
            ),
            [("islam", "Islam"), ("christianity", "Christianity")],
        )


if __name__ == "__main__":
    unittest.main()
