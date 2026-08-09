"""Compatibility facade for the split attention implementation."""

from __future__ import annotations

import sys
import types

from . import patches as _patches
from . import sparse as _sparse
from . import stable as _stable


for _module in (_stable, _sparse, _patches):
    for _name, _value in vars(_module).items():
        if not _name.startswith("__"):
            globals()[_name] = _value


class _AttentionFacade(types.ModuleType):
    """Keep legacy monkey-patching coherent across the split modules."""

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        for module in (_stable, _sparse, _patches):
            if hasattr(module, name):
                setattr(module, name, value)


sys.modules[__name__].__class__ = _AttentionFacade

del _module, _name, _value
