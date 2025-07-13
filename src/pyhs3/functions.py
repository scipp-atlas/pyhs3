"""
HS3 Functions implementation.

Provides classes for handling HS3 functions including product functions,
generic functions with mathematical expressions, and interpolation functions.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import pytensor.tensor as pt

from pyhs3 import typing as T
from pyhs3.generic_parse import analyze_sympy_expr, parse_expression, sympy_to_pytensor

log = logging.getLogger(__name__)


class Function:
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

    def expression(self, context: dict[str, T.TensorVar]) -> T.TensorVar:
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
    def from_dict(cls, config: dict[str, Any]) -> Function:
        """Create a Function instance from dictionary configuration."""
        raise NotImplementedError


class ProductFunction(Function):
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


class GenericFunction(Function):
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


class InterpolationFunction(Function):
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
        vars: list[str],
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
            vars: Variable names this function depends on
            **kwargs: Additional interpolation-specific parameters
        """
        # Use vars as the parameters this function depends on
        super().__init__(name=name, kind="interpolation", parameters=vars, **kwargs)
        self.high = high
        self.low = low
        self.nom = nom
        self.interpolationCodes = interpolationCodes
        self.positiveDefinite = positiveDefinite
        self.vars = vars

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
            vars=config["vars"],
        )

    def expression(self, _context: dict[str, T.TensorVar]) -> T.TensorVar:
        """Placeholder implementation - return constant for now."""
        log.warning(
            "Interpolation function %s not fully implemented, returning 1.0", self.name
        )
        return cast(T.TensorVar, pt.constant(1.0))


registered_functions: dict[str, type[Function]] = {
    "product": ProductFunction,
    "generic_function": GenericFunction,
    "interpolation": InterpolationFunction,
}


class FunctionSet:
    """Collection of HS3 functions."""

    def __init__(self, functions: list[dict[str, Any]]) -> None:
        """
        Collection of functions that compute parameter values.

        Args:
            functions: List of function configurations from HS3 spec
        """
        self.functions: dict[str, Function] = {}

        for func_config in functions:
            try:
                func_type = func_config["type"]
                the_func = registered_functions.get(func_type, Function)
                if the_func is Function:
                    msg = f"Unknown function type: {func_type}"
                    raise ValueError(msg)
                func = the_func.from_dict(
                    {k: v for k, v in func_config.items() if k != "type"}
                )
                self.functions[func.name] = func
            except Exception as exc:
                func_name = func_config.get("name", "unknown")
                func_type = func_config.get("type", "unknown")
                log.warning(
                    "Failed to create function %s of type %s: %s",
                    func_name,
                    func_type,
                    exc,
                )

    def __getitem__(self, item: str) -> Function:
        return self.functions[item]

    def __contains__(self, item: str) -> bool:
        return item in self.functions

    def __len__(self) -> int:
        return len(self.functions)
