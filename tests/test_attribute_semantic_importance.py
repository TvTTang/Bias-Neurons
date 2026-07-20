import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "attribute_semantic_importance.py"
)
SPEC = importlib.util.spec_from_file_location(
    "attribute_semantic_importance", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    mask_token = "[MASK]"
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    pad_token_id = 0
    vocabulary = {
        "[PAD]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[MASK]": 3,
        "plain": 4,
        "word": 5,
    }

    def convert_tokens_to_ids(self, value):
        if isinstance(value, list):
            return [self.vocabulary[token] for token in value]
        return self.vocabulary[value]


class AttributeSemanticImportanceTest(unittest.TestCase):
    def test_model_inputs_preserve_mask_and_target(self):
        tokenizer = FakeTokenizer()
        record = {
            "masked_tokens": ["plain", "[MASK]"],
            "mask_index": 1,
            "target_token": "word",
            "target_id": 5,
        }

        input_ids, attention, token_types, position, target = (
            MODULE.make_model_inputs(record, tokenizer, max_seq_length=7)
        )

        self.assertEqual(input_ids, [1, 4, 3, 2, 0, 0, 0])
        self.assertEqual(attention, [1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(token_types, [0] * 7)
        self.assertEqual(position, 2)
        self.assertEqual(target, 5)

    def test_invalid_target_mapping_is_rejected(self):
        tokenizer = FakeTokenizer()
        record = {
            "masked_tokens": ["[MASK]"],
            "mask_index": 0,
            "target_token": "word",
            "target_id": 4,
        }
        with self.assertRaisesRegex(ValueError, "disagree"):
            MODULE.validate_record(record, tokenizer)

    def test_rank_rows_uses_mean_absolute_ig(self):
        rows = MODULE.rank_rows(
            signed_sums=[-4.0, 1.0, 2.0, 0.0],
            abs_sums=[4.0, 1.0, 2.0, 0.0],
            sum_squares=[8.0, 0.5, 2.0, 0.0],
            count=2,
            neurons_per_layer=2,
        )

        self.assertEqual(rows[0]["semantic_rank"], 1)
        self.assertEqual(rows[2]["semantic_rank"], 2)
        self.assertEqual(rows[0]["semantic_mean_abs_ig"], 2.0)
        self.assertEqual(rows[0]["semantic_mean_signed_ig"], -2.0)


if __name__ == "__main__":
    unittest.main()
