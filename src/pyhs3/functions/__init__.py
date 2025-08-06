"""
HS3 Functions implementation.

Provides classes for handling HS3 functions including product functions,
generic functions with mathematical expressions, and interpolation functions.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Annotated, Any

from pydantic import (
    Field,
    PrivateAttr,
    RootModel,
)

from pyhs3.exceptions import custom_error_msg
from pyhs3.functions import standard
from pyhs3.functions.core import Function

log = logging.getLogger(__name__)

SumFunction = standard.SumFunction
ProductFunction = standard.ProductFunction
GenericFunction = standard.GenericFunction
InterpolationFunction = standard.InterpolationFunction
ProcessNormalizationFunction = standard.ProcessNormalizationFunction
CMSAsymPowFunction = standard.CMSAsymPowFunction
HistogramFunction = standard.HistogramFunction
RooRecursiveFractionFunction = standard.RooRecursiveFractionFunction


# Define the union type for all function configurations
FunctionConfig = (
    SumFunction
    | ProductFunction
    | GenericFunction
    | InterpolationFunction
    | ProcessNormalizationFunction
    | CMSAsymPowFunction
    | HistogramFunction
    | RooRecursiveFractionFunction
)

registered_functions: dict[str, type[Function]] = {
    "sum": SumFunction,
    "product": ProductFunction,
    "generic_function": GenericFunction,
    "interpolation": InterpolationFunction,
    "CMS::process_normalization": ProcessNormalizationFunction,
    "CMS::asympow": CMSAsymPowFunction,
    "histogram": HistogramFunction,
    "roorecursivefraction_dist": RooRecursiveFractionFunction,
}

# Type alias for all function types using discriminated union
FunctionType = Annotated[
    SumFunction
    | ProductFunction
    | GenericFunction
    | InterpolationFunction
    | ProcessNormalizationFunction
    | CMSAsymPowFunction
    | HistogramFunction
    | RooRecursiveFractionFunction,
    Field(discriminator="type"),
]


class Functions(RootModel[list[FunctionType]]):
    """
    Collection of HS3 functions for parameter computation.

    Manages a set of function instances that compute parameter values
    based on other parameters. Functions can be products, generic
    mathematical expressions, or interpolation functions.

    Provides dict-like access to functions by name and handles
    function creation from configuration dictionaries.

    Attributes:
        funcs: Mapping from function names to Function instances.
    """

    root: Annotated[
        list[FunctionType],
        custom_error_msg(
            {
                "union_tag_invalid": "Unknown function type '{tag}' does not match any of the expected functions: {expected_tags}"
            }
        ),
    ] = Field(default_factory=list)
    _map: dict[str, Function] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any, /) -> None:
        """Initialize computed collections after Pydantic validation."""
        self._map = {func.name: func for func in self.root}

    def __getitem__(self, item: str) -> Function:
        return self._map[item]

    def __contains__(self, item: str) -> bool:
        return item in self._map

    def __iter__(self) -> Iterator[Function]:  # type: ignore[override]  # https://github.com/pydantic/pydantic/issues/8872
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)
