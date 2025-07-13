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
    """

    type: Literal["interpolation"]
    name: str
    high: list[str]
    low: list[str]
    nom: list[str]
    interpolationCodes: list[str]
    positiveDefinite: bool
    vars: list[str]


Function = Union[
    ProductFunction,
    GenericFunction,
    InterpolationFunction,
]
