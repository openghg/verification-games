"""Metadata helpers for verification-games derived products."""

from __future__ import annotations

from datetime import UTC, datetime
import getpass
import socket


def append_history(attrs: dict, entry: str) -> dict:
    """Append a CF-style history entry to an attribute dictionary."""
    now = datetime.now(UTC)
    user = getpass.getuser()
    host = socket.gethostname()
    line = f"{now} {user}@{host}: {entry}"

    history = attrs.get("history", "")
    history = history.rstrip() + "\n" + line if history else line

    new_attrs = attrs.copy()
    new_attrs["history"] = history
    return new_attrs
