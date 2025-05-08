"""
typing parameter_point
"""

from __future__ import annotations

from typing import TypedDict


class Parameter(TypedDict):
    """
    Parameter
    """

    name: str
    value: float


class ParameterPoint(TypedDict):
    """
    ParameterPoint
    """

    name: str
    parameters: list[Parameter]
