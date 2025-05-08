"""
typing domain
"""

from __future__ import annotations

from typing import Literal, TypedDict, Union

from pyhs3.typing_compat import TypeAlias


class Axis(TypedDict):
    """
    Axis
    """

    name: str
    min: float
    max: float


class ProductDomain(TypedDict):
    """
    ProductDomain
    """

    type: Literal["product_domain"]
    name: str
    axes: list[Axis]


Domain: TypeAlias = Union[ProductDomain]
