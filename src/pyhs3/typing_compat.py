"""
Typing helpers.
"""

from __future__ import annotations

import sys
from typing import Annotated, TypeAlias

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired


__all__ = (
    "Annotated",
    "NotRequired",
    "TypeAlias",
)
