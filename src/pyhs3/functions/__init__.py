"""
HS3 Functions implementation.

Provides classes for handling HS3 functions including product functions,
generic functions with mathematical expressions, and interpolation functions.
"""

from __future__ import annotations

import logging
from typing import Annotated

from pydantic import Field, SerializeAsAny

from pyhs3.collections import NamedCollection, dispatch_on_type
from pyhs3.exceptions import custom_error_msg
from pyhs3.functions import core, standard
from pyhs3.functions.core import Function

log = logging.getLogger(__name__)

BatchableFunction = core.BatchableFunction
SumFunction = standard.SumFunction
ProductFunction = standard.ProductFunction
GenericFunction = standard.GenericFunction
InterpolationFunction = standard.InterpolationFunction
ProcessNormalizationFunction = standard.ProcessNormalizationFunction
CMSAsymPowFunction = standard.CMSAsymPowFunction
HistogramFunction = standard.HistogramFunction
RooRecursiveFractionFunction = standard.RooRecursiveFractionFunction


# Combine all function registries
registered_functions: dict[str, type[Function]] = {
    **standard.functions,
}

# Any registered function type, looked up by ``type`` when validated; see
# :func:`~pyhs3.collections.dispatch_on_type`. SerializeAsAny dumps each item
# with its own class's fields rather than only the base class's.
FunctionType = Annotated[
    SerializeAsAny[Function], dispatch_on_type(Function, registered_functions)
]


class Functions(NamedCollection[FunctionType]):
    """
    Collection of HS3 functions for parameter computation.

    Manages a set of function instances that compute parameter values
    based on other parameters. Functions can be products, generic
    mathematical expressions, or interpolation functions.

    Provides dict-like access to functions by name and handles
    function creation from configuration dictionaries.
    """

    _enforce_unique_names: bool = True

    root: Annotated[
        list[FunctionType],
        custom_error_msg(
            {
                "union_tag_invalid": "Unknown function type '{tag}' does not match any of the expected functions: {expected_tags}"
            }
        ),
    ] = Field(default_factory=list)
