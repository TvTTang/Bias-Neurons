import ast
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "bias_neuron_src"
    / "custom_bert_bias.py"
)


def load_scale_resolver():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "resolve_intervention_scale"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"math": __import__("math")}
    exec(compile(module, str(MODULE_PATH), "exec"), namespace)
    return namespace["resolve_intervention_scale"]


RESOLVE = load_scale_resolver()


class CustomBertBiasInterventionTest(unittest.TestCase):
    def test_legacy_operations_keep_their_original_scales(self):
        self.assertEqual(RESOLVE("remove"), 0.0)
        self.assertEqual(RESOLVE("enhance"), 2.0)

    def test_arbitrary_non_negative_scale_is_supported(self):
        self.assertEqual(RESOLVE("scale", 0.25), 0.25)
        self.assertEqual(RESOLVE("scale", 1.5), 1.5)

    def test_invalid_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            RESOLVE("scale", -0.1)
        with self.assertRaises(ValueError):
            RESOLVE("scale", float("nan"))


if __name__ == "__main__":
    unittest.main()
