"""
Mathematical Distribution implementations.

Provides classes for handling mathematical distributions defined by
mathematical formulas, polynomials, and custom expressions. These distributions
allow for flexible modeling using symbolic mathematical expressions.
"""

from __future__ import annotations

import logging
from typing import Literal, cast

import pytensor.tensor as pt
import sympy as sp
from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
)

from pyhs3.context import Context
from pyhs3.distributions.core import Distribution
from pyhs3.generic_parse import analyze_sympy_expr, parse_expression, sympy_to_pytensor
from pyhs3.typing.aliases import TensorVar

log = logging.getLogger(__name__)


class GenericDist(Distribution):
    """
    Generic distribution implementation.

    Evaluates custom mathematical expressions using SymPy parsing and
    PyTensor computation graphs.

    Parameters:
        name: Name of the distribution
        expression: Mathematical expression string to be evaluated

    Supported Functions:
        - Basic arithmetic: +, -, *, /, **
        - Trigonometric: sin, cos, tan
        - Exponential/Logarithmic: exp, log
        - Other: sqrt, abs

    Examples:
        Create a quadratic distribution:

        >>> dist = GenericDist(name="quadratic", expression="x**2 + 2*x + 1")

        Create a custom exponential with oscillation:

        >>> dist = GenericDist(name="exp_cos", expression="exp(-x**2/2) * cos(y)")

        Create a complex mathematical function:

        >>> dist = GenericDist(name="complex", expression="sin(x) + log(abs(y) + 1)")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, serialize_by_alias=True)

    type: Literal["generic_dist"] = Field(default="generic_dist", repr=False)
    expression_str: str = Field(alias="expression", repr=False)
    _sympy_expr: sp.Expr = PrivateAttr(default=None)
    _dependent_vars: list[str] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def setup_expression(self) -> GenericDist:
        """Parse and analyze the expression during initialization."""
        # Parse and analyze the expression during initialization
        self._sympy_expr = parse_expression(self.expression_str)

        # Analyze the expression to determine dependencies
        analysis = analyze_sympy_expr(self._sympy_expr)
        independent_vars = [str(symbol) for symbol in analysis["independent_vars"]]
        self._dependent_vars = [str(symbol) for symbol in analysis["dependent_vars"]]

        # Set parameters based on the analyzed expression
        self._parameters = {var: var for var in independent_vars}
        return self

    def expression(self, context: Context) -> TensorVar:
        """
        Evaluate the generic distribution using expression parsing.

        Args:
            context: Mapping of names to pytensor variables

        Returns:
            PyTensor expression representing the parsed mathematical expression

        Raises:
            ValueError: If the expression cannot be parsed or contains undefined variables
        """
        # Get the required variables using the parameters determined during initialization
        variables = [context[name] for name in self._parameters.values()]

        # Convert using the pre-parsed sympy expression
        result = sympy_to_pytensor(self._sympy_expr, variables)

        return cast(TensorVar, result)


class PolynomialDist(Distribution):
    r"""
    Polynomial probability distribution.

    Implements a polynomial probability density function as defined in ROOT's
    RooPolynomial and the HS3 specification:

    .. math::

        f(x; a_0, a_1, a_2, ...) = \frac{1}{\mathcal{M}} \sum_{i=0}^n a_i x^i = a_0 + a_1 x + a_2 x^2 + ...

    Parameters:
        x (str): Input variable name.
        coefficients (list[str]): Array of coefficient parameter names.

    Note:
        The degree of the polynomial is determined by the length of the coefficients array.
        ROOT uses a lowestOrder parameter to handle default coefficients, but for simplicity
        we require all coefficients to be explicitly specified.
    """

    type: Literal["polynomial_dist"] = "polynomial_dist"
    x: str | float | int
    coefficients: list[str]

    def expression(self, context: Context) -> TensorVar:
        """
        Builds a symbolic expression for the polynomial PDF.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of polynomial PDF.
        """
        x = context[self._parameters["x"]]

        # Build polynomial: sum(a_i * x^i)
        result = pt.constant(0.0)
        processed_coefficients = self.get_parameter_list(context, "coefficients")
        for i, coef in enumerate(processed_coefficients):
            result = result + coef if i == 0 else result + coef * x**i  # a_i * x^i

        return cast(TensorVar, result)


class BernsteinPolyDist(Distribution):
    r"""
    Bernstein polynomial probability distribution.

    Implements the Bernstein polynomial as defined in ROOT's RooBernstein.
    Used extensively for non-parametric fits and background modeling.

    .. math::

        f(x; c_0, c_1, ..., c_n) = \frac{1}{\mathcal{M}} \sum_{i=0}^n c_i B_{i,n}(x)

    where $B_{i,n}(x) = \binom{n}{i} x^i (1-x)^{n-i}$ are the Bernstein basis polynomials.

    Parameters:
        x (str): Input variable name (should be normalized to [0,1]).
        coefficients (list[str]): Array of coefficient parameter names.

    Note:
        The input variable is expected to be normalized to the [0,1] interval.
        The normalization to this interval is typically handled by the domain.
    """

    type: Literal["bernstein_poly_dist"] = "bernstein_poly_dist"
    x: str | float | int
    coefficients: list[str | float | int]

    def expression(self, context: Context) -> TensorVar:
        """
        Builds a symbolic expression for the Bernstein polynomial PDF.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of Bernstein polynomial PDF.
        """
        x = context[self._parameters["x"]]
        n = len(self.coefficients) - 1  # polynomial degree

        result = pt.constant(0.0)
        processed_coefficients = self.get_parameter_list(context, "coefficients")
        for i, coef in enumerate(processed_coefficients):
            # Bernstein basis polynomial: C(n,i) * x^i * (1-x)^(n-i)
            # Use pt.gammaln for log(C(n,i)) = log(Γ(n+1)) - log(Γ(i+1)) - log(Γ(n-i+1))
            log_binomial = pt.gammaln(n + 1) - pt.gammaln(i + 1) - pt.gammaln(n - i + 1)
            binomial = pt.exp(log_binomial)
            basis = binomial * (x**i) * ((1 - x) ** (n - i))
            result = result + coef * basis

        return cast(TensorVar, result)


# Registry of mathematical distributions
distributions: dict[str, type[Distribution]] = {
    "generic_dist": GenericDist,
    "polynomial_dist": PolynomialDist,
    "bernstein_poly_dist": BernsteinPolyDist,
}

# Define what should be exported from this module
__all__ = [
    "BernsteinPolyDist",
    "GenericDist",
    "PolynomialDist",
    "distributions",
]
