from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.nodes.logic import (  # noqa: E402
    IsInputPresent,
    LazyIfElse,
    is_value_present,
)


class IsInputPresentTest(unittest.TestCase):
    def test_schema_exposes_primary_fallback_and_two_outputs(self):
        schema = IsInputPresent.define_schema()
        self.assertEqual(schema.node_id, "TuringUtilsIsInputPresent")
        self.assertEqual(schema.inputs[0].id, "value")
        self.assertTrue(schema.inputs[0].optional)
        self.assertEqual(schema.inputs[0].io_type, "*")
        self.assertEqual(schema.inputs[1].id, "fallback")
        self.assertTrue(schema.inputs[1].optional)
        self.assertTrue(schema.inputs[1].lazy)
        self.assertEqual(schema.inputs[1].io_type, "*")
        self.assertEqual(schema.outputs[0].io_type, "BOOLEAN")
        self.assertEqual(schema.outputs[1].io_type, "*")

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
        self.assertEqual(IsInputPresent.execute().result, (False, None))

        value = torch.zeros(1)
        output = IsInputPresent.execute(value=value)
        self.assertEqual(output.result[0], True)
        self.assertIs(output.result[1], value)

    def test_empty_primary_selects_fallback(self):
        fallback = {"samples": torch.ones(1)}
        output = IsInputPresent.execute(value="", fallback=fallback)
        self.assertEqual(output.result[0], False)
        self.assertIs(output.result[1], fallback)

    def test_primary_value_skips_lazy_fallback(self):
        self.assertIsNone(
            IsInputPresent.check_lazy_status(value="primary", fallback=None)
        )

    def test_absent_primary_requests_only_connected_lazy_fallback(self):
        self.assertEqual(
            IsInputPresent.check_lazy_status(value=None, fallback=None),
            ["fallback"],
        )
        self.assertIsNone(IsInputPresent.check_lazy_status(value=None))


class LazyIfElseTest(unittest.TestCase):
    def test_schema_uses_optional_lazy_any_branch_inputs(self):
        schema = LazyIfElse.define_schema()
        self.assertEqual(schema.node_id, "TuringUtilsLazyIfElse")
        self.assertEqual(
            [item.id for item in schema.inputs],
            ["condition", "on_true", "on_false"],
        )
        self.assertFalse(schema.inputs[0].lazy)
        self.assertTrue(schema.inputs[1].lazy)
        self.assertTrue(schema.inputs[2].lazy)
        self.assertTrue(schema.inputs[1].optional)
        self.assertTrue(schema.inputs[2].optional)
        self.assertEqual(schema.inputs[1].io_type, "*")
        self.assertEqual(schema.inputs[2].io_type, "*")
        self.assertEqual(schema.outputs[0].io_type, "*")

    def test_lazy_status_requests_only_the_selected_missing_branch(self):
        self.assertEqual(
            LazyIfElse.check_lazy_status(True, on_true=None, on_false=None),
            ["on_true"],
        )
        self.assertEqual(
            LazyIfElse.check_lazy_status(False, on_true=None, on_false=None),
            ["on_false"],
        )
        self.assertIsNone(
            LazyIfElse.check_lazy_status(True, on_true="ready", on_false=None)
        )
        self.assertIsNone(
            LazyIfElse.check_lazy_status(False, on_true=None, on_false="ready")
        )

    def test_unconnected_selected_branch_needs_no_lazy_evaluation(self):
        self.assertIsNone(LazyIfElse.check_lazy_status(True))
        self.assertIsNone(LazyIfElse.check_lazy_status(False))
        self.assertEqual(LazyIfElse.execute(True).result, (None,))
        self.assertEqual(LazyIfElse.execute(False).result, (None,))

    def test_unselected_connected_branch_is_not_requested(self):
        self.assertIsNone(LazyIfElse.check_lazy_status(True, on_false=None))
        self.assertIsNone(LazyIfElse.check_lazy_status(False, on_true=None))

    def test_selected_explicit_none_is_forwarded_as_empty_output(self):
        self.assertEqual(
            LazyIfElse.execute(True, on_true=None, on_false="unused").result,
            (None,),
        )

    def test_execute_returns_selected_value_without_coercion(self):
        true_value = torch.zeros(1)
        false_value = {"samples": torch.ones(1)}
        self.assertIs(
            LazyIfElse.execute(True, true_value, false_value).result[0],
            true_value,
        )
        self.assertIs(
            LazyIfElse.execute(False, true_value, false_value).result[0],
            false_value,
        )


if __name__ == "__main__":
    unittest.main()
