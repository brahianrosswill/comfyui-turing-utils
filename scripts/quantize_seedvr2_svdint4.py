from __future__ import annotations

import sys


def main() -> int:
    print(
        "quantize_seedvr2_svdint4.py was retired with the native SeedVR2 runtime. "
        "SeedVR2 quantization must use a native calibration entrypoint so it does "
        "not depend on the removed external generation pipeline.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
