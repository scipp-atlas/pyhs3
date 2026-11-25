"""
Conversion between pyhf workspaces (HistFactory JSON, "HiFa") and HS3.

:func:`from_hifa` turns a pyhf workspace into a :class:`~pyhs3.Workspace`.
"""

from __future__ import annotations

from pyhs3.hifa._from_hifa import from_hifa

__all__ = ["from_hifa"]
