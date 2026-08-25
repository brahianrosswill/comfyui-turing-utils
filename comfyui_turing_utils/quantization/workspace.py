"""Pure workspace estimates shared by model memory planners."""

from __future__ import annotations


def int8_workspace_bytes(
    rows: int,
    output_channels: int,
    *,
    global_workspace_limit: int,
) -> int:
    if rows <= 0 or output_channels <= 0:
        return 0
    requested = int(rows) * int(output_channels) * 4
    fixed_workspace_compatible = int(output_channels) % 8 == 0
    return (
        0
        if fixed_workspace_compatible and requested >= int(global_workspace_limit)
        else requested
    )


def codebook_w4a8_workspace_bytes(
    input_channels: int,
    output_channels: int,
    *,
    chunk_rows: int,
) -> int:
    if input_channels <= 0 or output_channels <= 0:
        return 0
    return min(int(output_channels), int(chunk_rows)) * int(input_channels)


__all__ = ["codebook_w4a8_workspace_bytes", "int8_workspace_bytes"]
