"""Lazy imports for optional dependencies."""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def get_hist() -> Any:
    """
    Lazily import and return the hist module.

    Returns:
        The hist module

    Raises:
        ImportError: If hist is not installed
    """
    try:
        return importlib.import_module("hist")
    except ImportError as e:
        msg = (
            "Histogram visualization requires the 'hist' package. "
            "Install with: python -m pip install 'pyhs3[visualization]' or python -m pip install hist"
        )
        raise ImportError(msg) from e
