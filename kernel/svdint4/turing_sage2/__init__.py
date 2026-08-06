"""Compatibility alias for the bundled Turing Sage family.

New code should import :mod:`svdint4.turing_sage`. The old module name stays
available because releases before 0.6 exposed only the hybrid Sage2-derived
implementation from this path.
"""

from ..turing_sage import *  # noqa: F401,F403
from ..turing_sage import __all__
