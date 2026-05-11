"""Lightweight CF/Pint unit registry helpers.

OpenGHG exposes a customised CF unit registry as ``openghg.util._units.cf_ureg``.
Importing that path still executes ``openghg.__init__`` in normal Python import
semantics, so this module uses the same underlying ``cf_xarray`` registry and
activates ``pint-xarray`` without importing OpenGHG.
"""

from __future__ import annotations

import cf_xarray.units  # noqa: F401  # registers CF units/formatter
from cf_xarray.units import units as cf_ureg
import pint_xarray  # noqa: F401  # activates xarray's .pint accessor


__all__ = ["cf_ureg"]
