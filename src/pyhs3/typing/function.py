"""
typing function
"""

from __future__ import annotations

from typing import Literal, TypedDict


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


class SumFunction(TypedDict):
    """
    SumFunction
    """

    type: Literal["sum"]
    name: str
    summands: list[str]


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


class ProcessNormalization(TypedDict):
    """
    ProcessNormalization

    Implements process normalization function with systematic variations.
    Based on CMS Combine's ProcessNormalization class for HistFactory models.
    """

    type: Literal["ProcessNormalization"]
    name: str
    expression: str  # Expression for the function
    nominalValue: float  # Nominal normalization value
    thetaList: list[str]  # List of symmetric variation parameter names
    logKappa: list[float]  # List of symmetric log kappa values
    asymmThetaList: list[str]  # List of asymmetric variation parameter names
    logAsymmKappa: list[list[float]]  # List of asymmetric [low, high] log kappa values
    otherFactorList: list[str]  # List of additional multiplicative factors


Function = (
    ProductFunction
    | SumFunction
    | GenericFunction
    | InterpolationFunction
    | ProcessNormalization
)
