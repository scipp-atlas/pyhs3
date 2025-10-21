"""
HS3 Functions implementation.

Provides classes for handling HS3 functions including product functions,
generic functions with mathematical expressions, and interpolation functions.
"""

from __future__ import annotations

import logging
from typing import TypeVar

from pyhs3.base import Evaluable

log = logging.getLogger(__name__)


FuncT = TypeVar("FuncT", bound="Function")


class Function(Evaluable):
    """
    Base class for functions in HS3.

    Provides the foundation for all function implementations,
    handling parameter management and constant generation.

    Inherits parameter processing functionality from Evaluable.
    Subclasses must implement _expression() to define computation logic.
    """
