from .comfyui_turing_utils.bootstrap import bootstrap_builtin_integrations


bootstrap_builtin_integrations()

from .comfyui_turing_utils.registration import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
