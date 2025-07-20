"""
typing
"""

from __future__ import annotations

from typing import Any, TypedDict

from pyhs3.typing.aliases import TensorVar
from pyhs3.typing.distribution import Distribution
from pyhs3.typing.domain import Axis, Domain, ProductDomain
from pyhs3.typing.function import Function
from pyhs3.typing.metadata import Metadata
from pyhs3.typing.misc import Misc
from pyhs3.typing.parameter_point import Parameter, ParameterPoint
from pyhs3.typing_compat import NotRequired


class HS3Spec(TypedDict):
    """
    HS3Spec
    """

    distributions: NotRequired[list[Distribution]]
    functions: NotRequired[list[Function]]
    data: NotRequired[list[dict[str, Any]]]
    likelihoods: NotRequired[list[dict[str, Any]]]
    domains: NotRequired[list[ProductDomain]]
    parameter_points: NotRequired[list[ParameterPoint]]
    analyses: NotRequired[list[dict[str, Any]]]
    metadata: Metadata
    misc: NotRequired[Misc]


__all__ = (
    "Axis",
    "Distribution",
    "Domain",
    "Function",
    "HS3Spec",
    "Parameter",
    "ParameterPoint",
    "TensorVar",
)
