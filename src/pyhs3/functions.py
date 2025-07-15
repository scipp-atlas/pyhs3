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
from pyhs3.exceptions import UnknownInterpolationCodeError
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
        """
        Evaluate the product function.

        Args:
            context: Mapping of names to PyTensor variables.

        Returns:
            T.TensorVar: PyTensor expression representing the product of all factors.
        """
        if not self.factors:
            return pt.constant(1.0)

        result = context[self.factors[0]]
        for factor in self.factors[1:]:
            result = result * context[factor]

        return result


class GenericFunction(Function[TF.GenericFunction]):
    """
    Generic function with custom mathematical expression.

    Evaluates arbitrary mathematical expressions using SymPy parsing
    and PyTensor computation. Supports common mathematical operations
    including arithmetic, trigonometric, exponential, and logarithmic functions.

    The expression is parsed once during initialization and converted to
    a PyTensor computation graph for efficient evaluation.

    Parameters:
        name (str): Name of the function.
        expression (str): Mathematical expression string to evaluate.

    Examples:
        >>> func = GenericFunction(name="quadratic", expression="x**2 + 2*x + 1")
        >>> func = GenericFunction(name="sinusoid", expression="sin(x) * exp(-t)")
    """

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
        """
        Evaluate the generic function expression.

        Args:
            context: Mapping of names to PyTensor variables.

        Returns:
            T.TensorVar: PyTensor expression representing the parsed mathematical expression.
        """

        # Get required variables
        variables = [context[name] for name in self.parameters]

        # Convert using the pre-parsed sympy expression
        return sympy_to_pytensor(self.sympy_expr, variables)


class InterpolationFunction(Function[TF.InterpolationFunction]):
    r"""
    Piecewise interpolation function implementation.

    Implements ROOT's PiecewiseInterpolation logic to morph between nominal
    and variation distributions based on nuisance parameter values.
    Supports multiple interpolation codes (0-6) for different mathematical approaches.

    Mathematical Formulations:
        For **additive** interpolation modes (codes 0, 2, 3, 4):

        .. math::

            \text{result} = \text{nominal} + \sum_i I_i(\theta_i; \text{low}_i, \text{nominal}, \text{high}_i)

        For **multiplicative** interpolation modes (codes 1, 5, 6):

        .. math::

            \text{result} = \text{nominal} \times \prod_i [1 + I_i(\theta_i; \text{low}_i/\text{nominal}, 1, \text{high}_i/\text{nominal})]

    Interpolation Code Definitions:
        **Code 0** - Linear Interpolation/Extrapolation (Additive):

        .. math::

            I_0(\theta) = \begin{cases}
            \theta(\text{high} - \text{nom}) & \text{if } \theta \geq 0 \\
            \theta(\text{nom} - \text{low}) & \text{if } \theta < 0
            \end{cases}

        **Code 1** - Exponential Interpolation/Extrapolation (Multiplicative):

        .. math::

            I_1(\theta) = \begin{cases}
            \left(\frac{\text{high}}{\text{nom}}\right)^{\theta} - 1 & \text{if } \theta \geq 0 \\
            \left(\frac{\text{low}}{\text{nom}}\right)^{-\theta} - 1 & \text{if } \theta < 0
            \end{cases}

        **Code 2** - Exponential Interpolation + Linear Extrapolation (Additive):
        Uses :math:`\exp(\theta)` behavior for :math:`|\theta| \leq 1`, linear extrapolation for :math:`|\theta| > 1`
        with smooth transition at :math:`\theta = \pm 1`.

        **Code 3** - Exponential Interpolation + Different Linear Extrapolation (Additive):
        Uses :math:`\exp(\theta)` behavior for :math:`|\theta| \leq 1`, different linear extrapolation
        for :math:`|\theta| > 1` compared to code 2.

        **Code 4** - 6th Order Polynomial Interpolation + Linear Extrapolation (Additive):

        .. math::

            I_4(\theta) = \begin{cases}
            \text{linear extrapolation} & \text{if } |\theta| \geq 1 \\
            \theta \times (1 + \theta^2(-3 + \theta^2)/16) \times (\text{high} - \text{nom}) & \text{if } \theta \geq 0, |\theta| < 1
            \end{cases}

        **Code 5** - 6th Order Polynomial Interpolation + Exponential Extrapolation (Multiplicative):
        Uses exponential extrapolation for :math:`|\theta| \geq 1`, 6th order polynomial for :math:`|\theta| < 1`.
        Recommended for normalization factors.

        **Code 6** - 6th Order Polynomial Interpolation + Linear Extrapolation (Multiplicative):
        Uses linear extrapolation for :math:`|\theta| \geq 1`, 6th order polynomial for :math:`|\theta| < 1`.
        Recommended for normalization factors (no roots outside :math:`|\theta| < 1`).

    Args:
        name: Name of the function
        high: High variation parameter names
        low: Low variation parameter names
        nom: Nominal parameter name
        interpolationCodes: Interpolation method codes (0-6)
        positiveDefinite: Whether function should be positive definite
        parameters: Variable names this function depends on (nuisance parameters)

    Note:
        - At :math:`\theta_i = 0`, all codes return the nominal value
        - At :math:`\theta_i = \pm 1`, variations should match high/low values for appropriate codes
        - Polynomial codes (4,5,6) provide smoother interpolation with matching derivatives
        - Based on A.Bukin, Budker INP, Novosibirsk and ROOT's RooFit implementation
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

        Raises:
            UnknownInterpolationCodeError: If any interpolation code is not in range 0-6
        """
        super().__init__(name=name, kind="interpolation", parameters=parameters)

        # Validate interpolation codes at initialization
        valid_codes = {0, 1, 2, 3, 4, 5, 6}
        for code in interpolationCodes:
            if code not in valid_codes:
                msg = f"Unknown interpolation code {code} in function '{name}'. Valid codes are 0-6."
                raise UnknownInterpolationCodeError(msg)
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
        r"""
        Implement flexible interpolation for a single parameter.

        Based on ROOT's flexibleInterpSingle method with support for
        interpolation codes 0-6. This method computes the interpolation
        contribution :math:`I_i(\theta_i)` for a single nuisance parameter.

        Args:
            interp_code: Interpolation code (0-6) determining the mathematical approach
            low_val: Low variation value (used when :math:`\theta < 0`)
            high_val: High variation value (used when :math:`\theta \geq 0`)
            boundary: Boundary value for switching between interpolation and extrapolation (typically 1.0)
            nominal: Nominal value (baseline)
            param_val: Parameter value :math:`\theta` (nuisance parameter)

        Returns:
            Interpolated contribution :math:`I_i(\theta_i)` to be added (additive modes)
            or multiplied (multiplicative modes) with the result

        Note:
            The returned value interpretation depends on the interpolation code:
            - Codes 0,2,3,4: Direct additive contribution
            - Codes 1,5,6: Multiplicative factor (subtract 1 before use)
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

        # Code 6: Polynomial interpolation + linear extrapolation (multiplicative)
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
                    * (1 + param_val * param_val * (-3 + param_val * param_val) / 16),
                    param_val
                    * (ratio_low - 1.0)
                    * (1 + param_val * param_val * (-3 + param_val * param_val) / 16),
                ),
            ),
        )

    def expression(self, context: dict[str, T.TensorVar]) -> T.TensorVar:
        r"""
        Evaluate the interpolation function.

        Implements ROOT's PiecewiseInterpolation algorithm following the mathematical
        formulations described in the class docstring. The algorithm proceeds as:

        1. Start with nominal value: :math:`\text{result} = \text{nominal}`
        2. For each nuisance parameter :math:`\theta_i`, compute interpolation contribution :math:`I_i(\theta_i)`
        3. Combine contributions based on interpolation mode:
           - **Additive modes** (codes 0,2,3,4): :math:`\text{result} += I_i(\theta_i)`
           - **Multiplicative modes** (codes 1,5,6): :math:`\text{result} \times= (1 + I_i(\theta_i))`
        4. Apply positive definite constraint: :math:`\text{result} = \max(\text{result}, 0)` if requested

        Args:
            context: Mapping of names to pytensor variables containing:
                - Nominal parameter (referenced by `nom`)
                - High/low variation parameters (referenced by `high`/`low` lists)
                - Nuisance parameters (referenced by `parameters` list)

        Returns:
            PyTensor expression representing the interpolated result

        Note:
            The evaluation order ensures that all interpolation contributions are properly
            combined according to their mathematical modes before applying constraints.
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
    """
    Collection of HS3 functions for parameter computation.

    Manages a set of function instances that compute parameter values
    based on other parameters. Functions can be products, generic
    mathematical expressions, or interpolation functions.

    Provides dict-like access to functions by name and handles
    function creation from configuration dictionaries.

    Attributes:
        funcs (dict[str, Function[Any]]): Mapping from function names to Function instances.
    """

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
