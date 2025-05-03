from __future__ import annotations

from typing import Any, TypedDict

from pytensor.graph.basic import Apply
from pytensor.tensor.type import TensorType
from pytensor.tensor.variable import TensorVariable

from pyhs3.typing.distribution import Distribution
from pyhs3.typing.domain import Axis, Domain, ProductDomain
from pyhs3.typing.metadata import Metadata
from pyhs3.typing.misc import Misc
from pyhs3.typing.parameter_point import Parameter, ParameterPoint
from pyhs3.typing_compat import NotRequired, TypeAlias


class HS3Spec(TypedDict):
    distributions: NotRequired[list[Distribution]]
    functions: NotRequired[list[dict[str, Any]]]
    data: NotRequired[list[dict[str, Any]]]
    likelihoods: NotRequired[list[dict[str, Any]]]
    domains: NotRequired[list[ProductDomain]]
    parameter_points: NotRequired[list[ParameterPoint]]
    analyses: NotRequired[list[dict[str, Any]]]
    metadata: Metadata
    misc: NotRequired[Misc]


TensorVar: TypeAlias = TensorVariable[TensorType, Apply[Any]]

__all__ = (
    "Axis",
    "Distribution",
    "Domain",
    "HS3Spec",
    "Parameter",
    "ParameterPoint",
    "TensorVar",
)
