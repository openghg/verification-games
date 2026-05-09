"""Helpers for notebook use."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_src() -> Path:
    """Add the repository's ``src`` directory to ``sys.path`` if needed.

    Returns
    -------
    Path
        The absolute path to the ``src`` directory.

    Notes
    -----
    This is most useful for ad hoc notebooks or when using a kernel from a
    different environment. For longer-lived work, using ``pip install -e .`` or
    a dedicated project kernel is usually cleaner.
    """
    src = Path(__file__).resolve().parents[2]
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src
