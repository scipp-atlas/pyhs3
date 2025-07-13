"""
HS3 Functions implementation.

Provides classes for handling HS3 functions including product functions,
generic functions with mathematical expressions, and interpolation functions.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, Generic, TypeVar, cast

import pytensor.tensor as pt

from pyhs3 import typing as T
from pyhs3.generic_parse import analyze_sympy_expr, parse_expression, sympy_to_pytensor
from pyhs3.typing import function as TF

log = logging.getLogger(__name__)


FuncT = TypeVar("FuncT", bound="Function[T.Function]")
FuncConfigT = TypeVar("FuncConfigT", bound=T.Function)


class Function(Generic[FuncConfigT]):
    """Base class for HS3 functions."""

    def __init__(self, *, name: str, kind: str, parameters: list[str], **kwargs: Any):
        """
        Base class for functions that compute parameter values.

        Args:
            name: Name of the function
            kind: Type of the function (product, generic_function, interpolation)
            parameters: List of parameter/function names this function depends on
            **kwargs: Additional function-specific parameters
        """
        self.name = name
        self.kind = kind
        self.parameters = parameters
        self.kwargs = kwargs

    def expression(self, _: dict[str, T.TensorVar]) -> T.TensorVar:
        """
        Evaluate the function expression.

        Args:
            context: Mapping of names to pytensor variables

        Returns:
            PyTensor expression representing the function result
        """
        msg = f"Function type {self.kind} not implemented"
        raise NotImplementedError(msg)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Function[FuncConfigT]:
        """Create a Function instance from dictionary configuration."""
        raise NotImplementedError


class ProductFunction(Function[TF.ProductFunction]):
    """Product function that multiplies factors together."""

    def __init__(self, *, name: str, factors: list[str]):
        """
        Initialize a ProductFunction.

        Args:
            name: Name of the function
            factors: List of factor names to multiply together
        """
        # factors become the parameters this function depends on
        super().__init__(name=name, kind="product", parameters=factors)
        self.factors = factors

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ProductFunction:
        """Create a ProductFunction from dictionary configuration."""
        return cls(name=config["name"], factors=config["factors"])

    def expression(self, context: dict[str, T.TensorVar]) -> T.TensorVar:
        """Evaluate the product function."""
        if not self.factors:
            return pt.constant(1.0)

        result = context[self.factors[0]]
        for factor in self.factors[1:]:
            result = result * context[factor]

        return result


class GenericFunction(Function[TF.GenericFunction]):
    """Generic function with custom mathematical expression."""

    def __init__(self, *, name: str, expression: str):
        """
        Initialize a GenericFunction.

        Args:
            name: Name of the function
            expression: Mathematical expression string
        """
        self.expression_str = expression
        # Parse expression during initialization like GenericDist

        self.sympy_expr = parse_expression(expression)
        analysis = analyze_sympy_expr(self.sympy_expr)
        parameters = [str(symbol) for symbol in analysis["independent_vars"]]

        # Initialize parent with the parsed parameters
        super().__init__(name=name, kind="generic_function", parameters=parameters)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> GenericFunction:
        """Create a GenericFunction from dictionary configuration."""
        return cls(name=config["name"], expression=config["expression"])

    def expression(self, context: dict[str, T.TensorVar]) -> T.TensorVar:
        """Evaluate the generic function expression."""

        # Get required variables
        variables = [context[name] for name in self.parameters]

        # Convert using the pre-parsed sympy expression
        return sympy_to_pytensor(self.sympy_expr, variables)


class InterpolationFunction(Function[TF.InterpolationFunction]):
    """Interpolation function (placeholder implementation)."""

    def __init__(
        self,
        *,
        name: str,
        high: list[str],
        low: list[str],
        nom: list[str],
        interpolationCodes: list[str],
        positiveDefinite: bool,
        parameters: list[str],
        **kwargs: Any,
    ):
        """
        Initialize an InterpolationFunction.

        Args:
            name: Name of the function
            high: High variation parameter names
            low: Low variation parameter names
            nom: Nominal parameter names
            interpolationCodes: Interpolation method codes
            positiveDefinite: Whether function should be positive definite
            parameters: Variable names this function depends on
            **kwargs: Additional interpolation-specific parameters
        """
        super().__init__(
            name=name, kind="interpolation", parameters=parameters, **kwargs
        )
        self.high = high
        self.low = low
        self.nom = nom
        self.interpolationCodes = interpolationCodes
        self.positiveDefinite = positiveDefinite

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> InterpolationFunction:
        """Create an InterpolationFunction from dictionary configuration."""
        return cls(
            name=config["name"],
            high=config["high"],
            low=config["low"],
            nom=config["nom"],
            interpolationCodes=config["interpolationCodes"],
            positiveDefinite=config["positiveDefinite"],
            parameters=config["vars"],
        )

    def expression(self, _context: dict[str, T.TensorVar]) -> T.TensorVar:
        """Placeholder implementation - return constant for now."""
        log.warning(
            "Interpolation function %s not fully implemented, returning 1.0", self.name
        )
        return cast(T.TensorVar, pt.constant(1.0))


registered_functions: dict[str, type[Function[Any]]] = {
    "product": ProductFunction,
    "generic_function": GenericFunction,
    "interpolation": InterpolationFunction,
}


class FunctionSet:
    """Collection of HS3 functions."""

    def __init__(self, funcs: list[T.Function]) -> None:
        """
        Collection of functions that compute parameter values.

        Args:
            funcs: List of function configurations from HS3 spec
        """
        self.funcs: dict[str, Function[Any]] = {}
        for func_config in funcs:
            func_type = func_config["type"]
            the_func = registered_functions.get(func_type, Function)
            if the_func is Function:
                msg = f"Unknown function type: {func_type}"
                raise ValueError(msg)
            func = the_func.from_dict(
                {k: v for k, v in func_config.items() if k != "type"}
            )
            self.funcs[func.name] = func

    def __getitem__(self, item: str) -> Function[Any]:
        return self.funcs[item]

    def __contains__(self, item: str) -> bool:
        return item in self.funcs

    def __iter__(self) -> Iterator[Function[Any]]:
        return iter(self.funcs.values())

    def __len__(self) -> int:
        return len(self.funcs)
