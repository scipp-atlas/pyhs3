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
    mean: str
    sigma: str
    x: str


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


class CrystalDistribution(TypedDict):
    """
    CrystalDistribution
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


Distribution = Union[
    GaussianDistribution,
    MixtureDistribution,
    ProductDistribution,
    CrystalDistribution,
    GenericDistribution,
]
