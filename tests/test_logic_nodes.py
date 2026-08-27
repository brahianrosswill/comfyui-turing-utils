from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.nodes.logic import IsInputPresent, is_value_present  # noqa: E402


class IsInputPresentTest(unittest.TestCase):
    def test_schema_exposes_optional_any_input_and_boolean_output(self):
        schema = IsInputPresent.define_schema()
        self.assertEqual(schema.node_id, "TuringUtilsIsInputPresent")
        self.assertEqual(schema.inputs[0].id, "value")
        self.assertTrue(schema.inputs[0].optional)
        self.assertEqual(schema.inputs[0].io_type, "*")
        self.assertEqual(schema.outputs[0].io_type, "BOOLEAN")

    def test_missing_and_empty_values_are_absent(self):
        for value in (None, "", b"", [], (), {}, set(), torch.empty(0)):
            with self.subTest(value_type=type(value).__name__):
                self.assertFalse(is_value_present(value))

    def test_nonempty_values_are_present(self):
        for value in ("text", [None], {"samples": None}, torch.zeros(1)):
            with self.subTest(value_type=type(value).__name__):
                self.assertTrue(is_value_present(value))

    def test_false_and_zero_scalars_still_count_as_present(self):
        self.assertTrue(is_value_present(False))
        self.assertTrue(is_value_present(0))
        self.assertTrue(is_value_present(0.0))

    def test_execute_defaults_to_false_when_input_is_unconnected(self):
        self.assertEqual(IsInputPresent.execute().result, (False,))
        self.assertEqual(IsInputPresent.execute(value=torch.zeros(1)).result, (True,))


if __name__ == "__main__":
    unittest.main()
