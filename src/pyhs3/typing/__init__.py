"""
typing
"""

from __future__ import annotations

from typing import Any, TypedDict

from pyhs3.typing.aliases import TensorVar
from pyhs3.typing_compat import NotRequired


class HS3Spec(TypedDict):
    """
    HS3Spec
    """

    distributions: NotRequired[list[dict[str, Any]]]
    functions: NotRequired[list[dict[str, Any]]]
    data: NotRequired[list[dict[str, Any]]]
    likelihoods: NotRequired[list[dict[str, Any]]]
    domains: NotRequired[list[dict[str, Any]]]
    parameter_points: NotRequired[list[dict[str, Any]]]
    analyses: NotRequired[list[dict[str, Any]]]
    metadata: dict[str, Any]
    misc: NotRequired[dict[str, Any]]


__all__ = (
    "HS3Spec",
    "TensorVar",
)
