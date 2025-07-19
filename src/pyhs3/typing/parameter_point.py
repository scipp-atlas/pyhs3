"""
typing parameter_point
"""

from __future__ import annotations

from typing import TypedDict

import pyhs3.typing as T
from pyhs3.typing_compat import NotRequired


class Parameter(TypedDict):
    """
    Parameter
    """

    name: str
    const: NotRequired[bool]  # default: False
    value: float
    nbins: NotRequired[int]


class ParameterPoint(TypedDict):
    """
    ParameterPoint
    """

    name: str
    parameters: list[Parameter]
    kind: NotRequired[type[T.TensorVar]]  # defaults to pt.scalar
