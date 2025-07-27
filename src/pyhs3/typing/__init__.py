"""
typing
"""

from __future__ import annotations

from typing import Any, TypedDict

from pyhs3.distributions import DistributionConfig as Distribution
from pyhs3.domains import Axis
from pyhs3.domains import DomainConfig as Domain
from pyhs3.functions import FunctionConfig as Function
from pyhs3.parameter_points import ParameterPoint as Parameter
from pyhs3.parameter_points import ParameterSet as ParameterPoint
from pyhs3.typing.aliases import TensorVar
from pyhs3.typing.metadata import Metadata
from pyhs3.typing.misc import Misc
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
