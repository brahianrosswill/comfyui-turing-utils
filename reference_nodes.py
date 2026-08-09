"""Compatibility alias for :mod:`comfyui_turing_utils.media.references`."""

import sys
try:
    from .comfyui_turing_utils.media import references as _implementation
    from .comfyui_turing_utils.nodes.references import (
        OptionalResizeImageV2,
        ReferenceAudioHub,
        ReferenceImageHub,
        ReferenceVideoHub,
    )
except ImportError:
    from comfyui_turing_utils.media import references as _implementation
    from comfyui_turing_utils.nodes.references import (
        OptionalResizeImageV2,
        ReferenceAudioHub,
        ReferenceImageHub,
        ReferenceVideoHub,
    )

_implementation.OptionalResizeImageV2 = OptionalResizeImageV2
_implementation.ReferenceAudioHub = ReferenceAudioHub
_implementation.ReferenceImageHub = ReferenceImageHub
_implementation.ReferenceVideoHub = ReferenceVideoHub

sys.modules[__name__] = _implementation
