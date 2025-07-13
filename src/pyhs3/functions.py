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

    def __init__(self, *, name: str, kind: str, parameters: list[str]):
        """
        Base class for functions that compute parameter values.

        Args:
            name: Name of the function
            kind: Type of the function (product, generic_function, interpolation)
            parameters: List of parameter/function names this function depends on
        """
        self.name = name
        self.kind = kind
        self.parameters = parameters

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
    """
    Piecewise interpolation function implementation.

    Implements ROOT's PiecewiseInterpolation logic to morph between nominal
    and variation distributions based on nuisance parameter values.
    Supports multiple interpolation codes (0-6) for different mathematical approaches.
    """

    def __init__(
        self,
        *,
        name: str,
        high: list[str],
        low: list[str],
        nom: str,
        interpolationCodes: list[int],
        positiveDefinite: bool,
        parameters: list[str],
    ):
        """
        Initialize an InterpolationFunction.

        Args:
            name: Name of the function
            high: High variation parameter names
            low: Low variation parameter names
            nom: Nominal parameter name
            interpolationCodes: Interpolation method codes (0-6)
            positiveDefinite: Whether function should be positive definite
            parameters: Variable names this function depends on (nuisance parameters)
        """
        super().__init__(name=name, kind="interpolation", parameters=parameters)
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

    def _flexible_interp_single(
        self,
        interp_code: int,
        low_val: T.TensorVar,
        high_val: T.TensorVar,
        boundary: float,
        nominal: T.TensorVar,
        param_val: T.TensorVar,
    ) -> T.TensorVar:
        """
        Implement flexible interpolation for a single parameter.

        Based on ROOT's flexibleInterpSingle method with support for
        interpolation codes 0-6.

        Args:
            interp_code: Interpolation code (0-6)
            low_val: Low variation value
            high_val: High variation value
            boundary: Boundary value (typically 1.0)
            nominal: Nominal value
            param_val: Parameter value (theta)

        Returns:
            Interpolated contribution
        """
        # Codes 0, 2, 3, 4 are additive modes
        # Codes 1, 5, 6 are multiplicative modes

        if interp_code == 0:
            # Linear interpolation/extrapolation (additive)
            return cast(
                T.TensorVar,
                pt.switch(
                    param_val >= 0,
                    param_val * (high_val - nominal),
                    param_val * (nominal - low_val),
                ),
            )

        if interp_code == 1:
            # Exponential interpolation/extrapolation (multiplicative)
            ratio_high = high_val / nominal
            ratio_low = low_val / nominal
            return cast(
                T.TensorVar,
                pt.switch(
                    param_val >= 0,
                    cast(T.TensorVar, pt.power(ratio_high, param_val)) - 1.0,  # type: ignore[no-untyped-call]
                    cast(T.TensorVar, pt.power(ratio_low, -param_val)) - 1.0,  # type: ignore[no-untyped-call]
                ),
            )

        if interp_code == 2:
            # Exponential interpolation, linear extrapolation (additive)
            return cast(
                T.TensorVar,
                pt.switch(
                    pt.abs(param_val) <= boundary,
                    # Exponential interpolation for |theta| <= 1
                    pt.switch(
                        param_val >= 0,
                        (high_val - nominal) * (pt.exp(param_val) - 1),
                        (nominal - low_val) * (pt.exp(-param_val) - 1),
                    ),
                    # Linear extrapolation for |theta| > 1
                    pt.switch(
                        param_val >= 0,
                        (high_val - nominal)
                        * (
                            pt.exp(boundary)
                            - 1
                            + (param_val - boundary) * pt.exp(boundary)
                        ),
                        (nominal - low_val)
                        * (
                            pt.exp(boundary)
                            - 1
                            + (-param_val - boundary) * pt.exp(boundary)
                        ),
                    ),
                ),
            )

        if interp_code == 3:
            # Similar to code 2 but with different extrapolation
            return cast(
                T.TensorVar,
                pt.switch(
                    pt.abs(param_val) <= boundary,
                    # Exponential interpolation for |theta| <= 1
                    pt.switch(
                        param_val >= 0,
                        (high_val - nominal) * (pt.exp(param_val) - 1),
                        (nominal - low_val) * (pt.exp(-param_val) - 1),
                    ),
                    # Linear extrapolation for |theta| > 1
                    pt.switch(
                        param_val >= 0,
                        param_val * (high_val - nominal),
                        param_val * (nominal - low_val),
                    ),
                ),
            )

        if interp_code == 4:
            # Polynomial interpolation + linear extrapolation (additive)
            return cast(
                T.TensorVar,
                pt.switch(
                    pt.abs(param_val) >= boundary,
                    # Linear extrapolation for |theta| >= 1
                    pt.switch(
                        param_val >= 0,
                        param_val * (high_val - nominal),
                        param_val * (nominal - low_val),
                    ),
                    # 6th order polynomial interpolation for |theta| < 1
                    pt.switch(
                        param_val >= 0,
                        param_val
                        * (high_val - nominal)
                        * (
                            1
                            + param_val * param_val * (-3 + param_val * param_val) / 16
                        ),
                        param_val
                        * (nominal - low_val)
                        * (
                            1
                            + param_val * param_val * (-3 + param_val * param_val) / 16
                        ),
                    ),
                ),
            )

        if interp_code == 5:
            # Polynomial interpolation + exponential extrapolation (multiplicative)
            ratio_high = high_val / nominal
            ratio_low = low_val / nominal
            return cast(
                T.TensorVar,
                pt.switch(
                    pt.abs(param_val) >= boundary,
                    # Exponential extrapolation for |theta| >= 1
                    pt.switch(
                        param_val >= 0,
                        cast(T.TensorVar, pt.power(ratio_high, param_val)) - 1.0,  # type: ignore[no-untyped-call]
                        cast(T.TensorVar, pt.power(ratio_low, -param_val)) - 1.0,  # type: ignore[no-untyped-call]
                    ),
                    # 6th order polynomial interpolation for |theta| < 1
                    pt.switch(
                        param_val >= 0,
                        param_val
                        * (ratio_high - 1.0)
                        * (
                            1
                            + param_val * param_val * (-3 + param_val * param_val) / 16
                        ),
                        param_val
                        * (ratio_low - 1.0)
                        * (
                            1
                            + param_val * param_val * (-3 + param_val * param_val) / 16
                        ),
                    ),
                ),
            )

        if interp_code == 6:
            # Polynomial interpolation + linear extrapolation (multiplicative)
            ratio_high = high_val / nominal
            ratio_low = low_val / nominal
            return cast(
                T.TensorVar,
                pt.switch(
                    pt.abs(param_val) >= boundary,
                    # Linear extrapolation for |theta| >= 1
                    pt.switch(
                        param_val >= 0,
                        param_val * (ratio_high - 1.0),
                        param_val * (ratio_low - 1.0),
                    ),
                    # 6th order polynomial interpolation for |theta| < 1
                    pt.switch(
                        param_val >= 0,
                        param_val
                        * (ratio_high - 1.0)
                        * (
                            1
                            + param_val * param_val * (-3 + param_val * param_val) / 16
                        ),
                        param_val
                        * (ratio_low - 1.0)
                        * (
                            1
                            + param_val * param_val * (-3 + param_val * param_val) / 16
                        ),
                    ),
                ),
            )
        # Default to linear interpolation for unknown codes
        log.warning(
            "Unknown interpolation code %d, using linear interpolation", interp_code
        )
        return cast(
            T.TensorVar,
            pt.switch(
                param_val >= 0,
                param_val * (high_val - nominal),
                param_val * (nominal - low_val),
            ),
        )

    def expression(self, context: dict[str, T.TensorVar]) -> T.TensorVar:
        """
        Evaluate the interpolation function.

        Implements ROOT's PiecewiseInterpolation algorithm:
        1. Start with nominal value
        2. For each nuisance parameter, add interpolated contribution
        3. Apply positive definite constraint if requested

        Args:
            context: Mapping of names to pytensor variables

        Returns:
            PyTensor expression representing the interpolated result
        """
        # Start with nominal value
        nominal = context[self.nom]
        result = nominal

        # Apply interpolation for each nuisance parameter
        for i, var_name in enumerate(self.parameters):
            if (
                i >= len(self.high)
                or i >= len(self.low)
                or i >= len(self.interpolationCodes)
            ):
                log.warning(
                    "Parameter index %d exceeds variation lists for function %s",
                    i,
                    self.name,
                )
                continue

            param_val = context[var_name]
            low_val = context[self.low[i]]
            high_val = context[self.high[i]]
            interp_code = self.interpolationCodes[i]

            # Calculate interpolated contribution
            contribution = self._flexible_interp_single(
                interp_code=interp_code,
                low_val=low_val,
                high_val=high_val,
                boundary=1.0,
                nominal=nominal,
                param_val=param_val,
            )

            # Add contribution based on interpolation mode
            if interp_code in [0, 2, 3, 4]:  # Additive modes
                result = result + contribution
            else:  # Multiplicative modes (1, 5, 6)
                result = result * (1.0 + contribution)

        # Apply positive definite constraint if requested
        if self.positiveDefinite:
            result = pt.maximum(result, 0.0)

        return result


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
