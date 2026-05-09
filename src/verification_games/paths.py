"""Helpers for locating project directories.

These functions make notebook and script code less sensitive to the current
working directory.
"""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root.

    This file lives at ``src/<package>/paths.py``, so the repository root is
    three levels above this file.
    """
    return Path(__file__).resolve().parents[2]


def data_dir(*parts: str) -> Path:
    """Return a path inside the repository's ``data`` directory."""
    return repo_root().joinpath("data", *parts)


def notebooks_dir(*parts: str) -> Path:
    """Return a path inside the repository's ``notebooks`` directory."""
    return repo_root().joinpath("notebooks", *parts)
