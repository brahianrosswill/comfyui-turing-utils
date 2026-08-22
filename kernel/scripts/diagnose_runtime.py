#!/usr/bin/env python3
"""Print one JSON capability report without changing model state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
COMFY_ROOT = ROOT.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "kernel"), str(COMFY_ROOT)]

import torch

from comfyui_turing_utils.runtime import runtime_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(runtime_diagnostics(torch.device(args.device)), indent=2))


if __name__ == "__main__":
    main()
