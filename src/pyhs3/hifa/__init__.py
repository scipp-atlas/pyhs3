"""
Conversion between pyhf workspaces (HistFactory JSON, "HiFa") and HS3.

:func:`from_hifa` turns a pyhf workspace into a :class:`~pyhs3.Workspace`;
:func:`to_hifa` turns one back, with :func:`hifa_modifier_settings` giving the
interpolation codes pyhf's JSON cannot hold.
"""

from __future__ import annotations

from pyhs3.hifa._from_hifa import from_hifa
from pyhs3.hifa._to_hifa import hifa_modifier_settings, to_hifa

__all__ = ["from_hifa", "hifa_modifier_settings", "to_hifa"]
