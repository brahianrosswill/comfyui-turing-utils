# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

def init_sam2_hydra():
    """Compatibility no-op; the vendored builder reads OmegaConf files directly.

    This file is derived from Comfyui-SecNodes and modified by Turing Utils to
    avoid clearing or replacing ComfyUI's process-global Hydra state.
    """
    return None

# The upstream runtime called this function before constructing SAM2.  Keeping
# the symbol avoids a larger vendor diff while removing the global side effect.
