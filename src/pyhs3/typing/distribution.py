"""
typing distribution
"""

from __future__ import annotations

from typing import Literal, TypedDict, Union

from pyhs3.typing_compat import NotRequired


class GaussianDistribution(TypedDict):
    """
    GaussianDistribution
    """

    type: Literal["gaussian_dist"]
    name: str
    mean: str | float | int
    sigma: str | float | int
    x: str | float | int


class MixtureDistribution(TypedDict):
    """
    MixtureDistribution
    """

    type: Literal["mixture_dist"]
    name: str
    summands: list[str]
    coefficients: list[str]
    extended: NotRequired[bool]


class ProductDistribution(TypedDict):
    """
    ProductDistribution
    """

    type: Literal["product_dist"]
    name: str
    factors: list[str]


class CrystalBallDistribution(TypedDict):
    """
    CrystalBallDistribution
    """

    type: Literal["crystalball_doublesided_dist"]
    name: str
    alpha_L: str
    alpha_R: str
    m: str
    m0: str
    n_L: str
    n_R: str
    sigma_R: str
    sigma_L: str


class GenericDistribution(TypedDict):
    """
    GenericDistribution
    """

    type: Literal["generic_dist"]
    name: str
    expression: str


class PoissonDistribution(TypedDict):
    """
    PoissonDistribution
    """

    type: Literal["poisson_dist"]
    name: str
    mean: str | float | int
    x: str | float | int


Distribution = Union[
    GaussianDistribution,
    MixtureDistribution,
    ProductDistribution,
    CrystalBallDistribution,
    GenericDistribution,
    PoissonDistribution,
]
