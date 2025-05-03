from __future__ import annotations

from typing import Literal, TypedDict, Union

from pyhs3.typing_compat import NotRequired


class MixtureDistribution(TypedDict):
    type: Literal["mixture_dist"]
    name: str
    summands: list[str]
    coefficients: list[str]
    extended: NotRequired[bool]


class GaussianDistribution(TypedDict):
    type: Literal["gaussian_dist"]
    name: str
    mean: str
    sigma: str
    x: str


Distribution = Union[MixtureDistribution, GaussianDistribution]
