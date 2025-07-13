"""
typing function
"""

from __future__ import annotations

from typing import Literal, TypedDict, Union


class ProductFunction(TypedDict):
    """
    ProductFunction
    """

    type: Literal["product"]
    name: str
    factors: list[str]


class GenericFunction(TypedDict):
    """
    GenericFunction
    """

    type: Literal["generic_function"]
    name: str
    expression: str


class InterpolationFunction(TypedDict):
    """
    InterpolationFunction

    Implements piecewise interpolation between nominal and variation distributions.
    Based on ROOT's PiecewiseInterpolation class for HistFactory models.
    """

    type: Literal["interpolation"]
    name: str
    high: list[str]  # List of high variation parameter names
    low: list[str]  # List of low variation parameter names
    nom: str  # Single nominal parameter name
    interpolationCodes: list[int]  # List of interpolation codes (0-6)
    positiveDefinite: bool  # Whether result should be positive definite
    vars: list[str]  # List of nuisance parameter names


Function = Union[
    ProductFunction,
    GenericFunction,
    InterpolationFunction,
]
