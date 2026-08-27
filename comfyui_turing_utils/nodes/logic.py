"""Small workflow-control helpers."""

from __future__ import annotations

import torch
from comfy_api.latest import io


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
                "Zero and false scalar values still count as present."
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
            ],
            outputs=[io.Boolean.Output(display_name="present")],
        )

    @classmethod
    def execute(cls, value=None) -> io.NodeOutput:
        return io.NodeOutput(is_value_present(value))
