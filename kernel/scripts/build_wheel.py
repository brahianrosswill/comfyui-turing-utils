from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT, env=env)


def configure_cuda_home(env: dict[str, str]) -> Path | None:
    """Find NVCC in activated Conda CUDA layouts before importing PyTorch."""
    executable = "nvcc.exe" if platform.system() == "Windows" else "nvcc"
    roots: list[Path] = []
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        if env.get(variable):
            roots.append(Path(env[variable]))
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix)
        roots.append(prefix / "Library" if platform.system() == "Windows" else prefix)
    for root in roots:
        candidate = root / "bin" / executable
        if candidate.is_file():
            env.setdefault("CUDA_HOME", str(root.resolve()))
            return candidate.resolve()
    discovered = shutil.which("nvcc")
    return Path(discovered).resolve() if discovered else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a comfyui-turing-utils-kernel wheel.")
    parser.add_argument(
        "--arch-list",
        default=None,
        help="CUDA architectures to compile; defaults to all visible supported GPUs.",
    )
    parser.add_argument("--with-isolation", action="store_true", help="Use PEP 517 build isolation.")
    parser.add_argument("--skip-build-deps", action="store_true", help="Do not install pip/build/wheel/ninja.")
    args = parser.parse_args()

    env = os.environ.copy()
    if args.arch_list:
        env["COMFYUI_TURING_UTILS_ARCH_LIST"] = args.arch_list
    nvcc = configure_cuda_home(env)

    if platform.system() == "Windows":
        if shutil.which("cl") is None:
            print("warning: cl.exe was not found; run from an x64 Visual Studio Developer shell.", file=sys.stderr)
        env.setdefault("DISTUTILS_USE_SDK", "1")
        env.setdefault("MSSdk", "1")

    if nvcc is None:
        print("warning: nvcc was not found on PATH; CUDA_HOME must point at a CUDA toolkit.", file=sys.stderr)

    if not args.skip_build_deps:
        run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "build", "wheel", "ninja"], env)

    shutil.rmtree(ROOT / "build", ignore_errors=True)
    shutil.rmtree(ROOT / "comfyui_turing_utils_kernel.egg-info", ignore_errors=True)

    cmd = [sys.executable, "-m", "build", "--wheel"]
    if not args.with_isolation:
        cmd.extend(["--no-isolation", "--skip-dependency-check"])
    run(cmd, env)


if __name__ == "__main__":
    main()
