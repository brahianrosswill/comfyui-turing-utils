from __future__ import annotations


SVDINT4_FORMAT = "svdint4"
SUPPORTED_ARCHITECTURES = frozenset({"wan"})


def validate_svdint4_metadata(metadata: dict[str, str], source: object = "model") -> str:
    fmt = metadata.get("format")
    if fmt != SVDINT4_FORMAT:
        raise ValueError(f"{source} is not an SVDInt4 file: format={fmt!r}; expected {SVDINT4_FORMAT!r}")
    architecture = metadata.get("architecture")
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"{source} has unsupported SVDInt4 architecture={architecture!r}; "
            f"expected one of {sorted(SUPPORTED_ARCHITECTURES)}"
        )
    return architecture
