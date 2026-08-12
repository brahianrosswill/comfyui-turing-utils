# Experimental Turing Sage1/Sage2 variants

The SM75 Sage1 and Sage2 adaptations are **unstable research code**. Local
Turing tests produced severe block artefacts and black flicker, and Sage2 did
not provide a useful speedup. They are therefore intentionally absent from:

- the ComfyUI loader choices;
- the `comfyui_turing_utils_kernel.turing_sage` public API;
- the default `comfyui-turing-utils-kernel` Python package; and
- the default CUDA extension bindings and template instantiations.

The complete implementation, precision comparison tool, validation cases and
benchmark breakdowns are preserved in git checkpoint `2b63f13`. Use a detached
worktree for experiments so an experimental build cannot replace the production
kernel accidentally:

```bash
git worktree add ../turing-sage-experiments 2b63f13
cd ../turing-sage-experiments/kernel
COMFYUI_TURING_UTILS_ARCH_LIST=7.5 python -m pip install .
```

That checkpoint exposes `sage_`, `sage1`, and `sage2`. The production branch
renames the proven `sage_` implementation to `sage` and builds only that path.
The research CUDA is intentionally absent from the production `csrc` tree. The
checkpoint above is the sole reproducible source so retired kernels cannot
silently return to release builds through a compile flag.
