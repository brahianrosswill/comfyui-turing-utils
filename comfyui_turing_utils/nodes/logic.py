"""Small workflow-control helpers."""

from __future__ import annotations

import torch
from comfy_api.latest import io


_MISSING = object()


def is_value_present(value=None) -> bool:
    """Return whether an optional workflow value exists and is non-empty.

    Presence is intentionally different from Python truthiness: connected
    scalar values such as ``0`` and ``False`` are valid values.  Only a missing
    value, ``None``, or an object with no elements is considered absent.
    """
    if value is None:
        return False
    if torch.is_tensor(value):
        return value.numel() > 0
    try:
        return len(value) > 0
    except (TypeError, AttributeError):
        return True


class IsInputPresent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsIsInputPresent",
            display_name="Is Input Present",
            category="Turing Utils/logic",
            description=(
                "Returns true when the optional input is connected and non-empty. "
                "The value output forwards the primary input when present, otherwise "
                "it lazily evaluates and forwards the optional fallback. Zero and "
                "false scalar values still count as present."
            ),
            inputs=[
                io.AnyType.Input(
                    "value",
                    optional=True,
                    tooltip=(
                        "Connect any value. An unconnected input, None, an empty "
                        "string/container, or a zero-element tensor returns false."
                    ),
                ),
                io.AnyType.Input(
                    "fallback",
                    optional=True,
                    lazy=True,
                    tooltip=(
                        "Returned only when value is absent or empty. Its upstream "
                        "branch is evaluated lazily."
                    ),
                ),
            ],
            outputs=[
                io.Boolean.Output(display_name="present"),
                io.AnyType.Output(display_name="value"),
            ],
        )

    @classmethod
    def check_lazy_status(cls, value=None, fallback=_MISSING):
        if is_value_present(value) or fallback is _MISSING:
            return None
        if fallback is None:
            return ["fallback"]

    @classmethod
    def execute(cls, value=None, fallback=_MISSING) -> io.NodeOutput:
        present = is_value_present(value)
        selected = value if present else fallback
        return io.NodeOutput(
            present,
            None if selected is _MISSING else selected,
        )


class LazyIfElse(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsLazyIfElse",
            display_name="Lazy If / Else",
            category="Turing Utils/logic",
            description=(
                "Returns the selected branch and asks ComfyUI to evaluate only that "
                "lazy input. An unselected branch is skipped unless another output "
                "in the workflow also requires it."
            ),
            search_aliases=["if", "else", "switch", "branch", "lazy"],
            inputs=[
                io.Boolean.Input("condition"),
                io.AnyType.Input(
                    "on_true",
                    lazy=True,
                    optional=True,
                    tooltip="Evaluated only when condition is true.",
                ),
                io.AnyType.Input(
                    "on_false",
                    lazy=True,
                    optional=True,
                    tooltip="Evaluated only when condition is false.",
                ),
            ],
            outputs=[io.AnyType.Output(display_name="value")],
        )

    @classmethod
    def check_lazy_status(
        cls, condition, on_true=_MISSING, on_false=_MISSING
    ):
        selected = on_true if condition else on_false
        if selected is _MISSING:
            return None
        if condition and selected is None:
            return ["on_true"]
        if not condition and selected is None:
            return ["on_false"]

    @classmethod
    def execute(
        cls, condition, on_true=_MISSING, on_false=_MISSING
    ) -> io.NodeOutput:
        selected = on_true if condition else on_false
        return io.NodeOutput(None if selected is _MISSING else selected)
