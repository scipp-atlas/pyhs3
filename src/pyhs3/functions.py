"""
HS3 Functions implementation.

Provides classes for handling HS3 functions including product functions,
generic functions with mathematical expressions, and interpolation functions.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, Literal, TypeVar, cast

import pytensor.tensor as pt
import sympy as sp
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pyhs3.exceptions import UnknownInterpolationCodeError
from pyhs3.generic_parse import analyze_sympy_expr, parse_expression, sympy_to_pytensor
from pyhs3.typing.aliases import TensorVar

log = logging.getLogger(__name__)


FuncT = TypeVar("FuncT", bound="Function")


class Function(BaseModel):
    """Base class for HS3 functions."""

    model_config = ConfigDict(serialize_by_alias=True)

    name: str
    type: str
    parameters: dict[str, str] = Field(default_factory=dict, exclude=True)
    constants_values: dict[str, float | int] = Field(default_factory=dict, exclude=True)

    @property
    def constants(self) -> dict[str, TensorVar]:
        """Convert stored numeric constants to PyTensor constants."""
        return {
            name: cast(TensorVar, pt.constant(value))
            for name, value in self.constants_values.items()
        }

    def expression(self, _: dict[str, TensorVar]) -> TensorVar:
        """
        Evaluate the function expression.

        Args:
            context: Mapping of names to pytensor variables

        Returns:
            PyTensor expression representing the function result
        """
        msg = f"Function type {self.type} not implemented"
        raise NotImplementedError(msg)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Function:
        """Create a Function instance from dictionary configuration."""
        raise NotImplementedError


class SumFunction(Function):
    """Sum function that adds summands together."""

    type: Literal["sum"] = "sum"
    summands: list[str]

    @model_validator(mode="after")
    def process_parameters(self) -> SumFunction:
        """Build the parameters dict from summands."""
        self.parameters = {name: name for name in self.summands}
        return self

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> SumFunction:
        """Create a SumFunction from dictionary configuration."""
        return cls(**config)

    def expression(self, context: dict[str, TensorVar]) -> TensorVar:
        """
        Evaluate the sum function.

        Args:
            context: Mapping of names to PyTensor variables.

        Returns:
            TensorVar: PyTensor expression representing the sum of all summands.
        """
        if not self.summands:
            return pt.constant(0.0)

        result = context[self.summands[0]]
        for summand in self.summands[1:]:
            result = result + context[summand]

        return result


class ProductFunction(Function):
    """Product function that multiplies factors together."""

    type: Literal["product"] = "product"
    factors: list[str]

    @model_validator(mode="after")
    def process_parameters(self) -> ProductFunction:
        """Build the parameters dict from factors."""
        self.parameters = {name: name for name in self.factors}
        return self

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ProductFunction:
        """Create a ProductFunction from dictionary configuration."""
        return cls(**config)

    def expression(self, context: dict[str, TensorVar]) -> TensorVar:
        """
        Evaluate the product function.

        Args:
            context: Mapping of names to PyTensor variables.

        Returns:
            TensorVar: PyTensor expression representing the product of all factors.
        """
        if not self.factors:
            return pt.constant(1.0)

        result = context[self.factors[0]]
        for factor in self.factors[1:]:
            result = result * context[factor]

        return result


class GenericFunction(Function):
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

    model_config = ConfigDict(arbitrary_types_allowed=True, serialize_by_alias=True)

    type: Literal["generic_function"] = "generic_function"
    expression_str: str = Field(alias="expression")
    sympy_expr: sp.Expr = Field(default=None, exclude=True)
    dependent_vars: list[str] = Field(default_factory=list, exclude=True)

    @model_validator(mode="after")
    def setup_expression(self) -> GenericFunction:
        """Parse and analyze the expression during initialization."""
        # Parse and analyze the expression during initialization
        self.sympy_expr = parse_expression(self.expression_str)

        # Analyze the expression to determine dependencies
        analysis = analyze_sympy_expr(self.sympy_expr)
        independent_vars = [str(symbol) for symbol in analysis["independent_vars"]]
        self.dependent_vars = [str(symbol) for symbol in analysis["dependent_vars"]]

        # Set parameters based on the analyzed expression
        self.parameters = {var: var for var in independent_vars}
        return self

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> GenericFunction:
        """Create a GenericFunction from dictionary configuration."""
        return cls(**config)

    def expression(self, context: dict[str, TensorVar]) -> TensorVar:
        """
        Evaluate the generic function expression.

        Args:
            context: Mapping of names to PyTensor variables.

        Returns:
            TensorVar: PyTensor expression representing the parsed mathematical expression.
        """
        # Get required variables using the parameters determined during initialization
        variables = [context[name] for name in self.parameters.values()]

        # Convert using the pre-parsed sympy expression
        return sympy_to_pytensor(self.sympy_expr, variables)


class InterpolationFunction(Function):
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

    Parameters:
        name: Name of the function
        high: High variation parameter names
        low: Low variation parameter names
        nom: Nominal parameter name
        interpolationCodes: Interpolation method codes (0-6)
        positiveDefinite: Whether function should be positive definite
        vars: Variable names this function depends on (nuisance parameters)
    """

    type: Literal["interpolation"] = "interpolation"
    high: list[str]
    low: list[str]
    nom: str
    interpolationCodes: list[int]
    positiveDefinite: bool
    vars: list[str]

    @model_validator(mode="after")
    def process_parameters(self) -> InterpolationFunction:
        """Build the parameters dict and validate interpolation codes."""

        # Validate interpolation codes
        valid_codes = {0, 1, 2, 3, 4, 5, 6}
        for code in self.interpolationCodes:
            if code not in valid_codes:
                msg = f"Unknown interpolation code {code} in function '{self.name}'. Valid codes are 0-6."
                raise UnknownInterpolationCodeError(msg)

        # Build parameters dict - all high, low, nom, and vars parameters
        all_params = [*self.high, *self.low, self.nom, *self.vars]
        self.parameters = {name: name for name in all_params}
        return self

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> InterpolationFunction:
        """Create an InterpolationFunction from dictionary configuration."""
        return cls(**config)

    def expression(self, context: dict[str, TensorVar]) -> TensorVar:
        """
        Evaluate the interpolation function.

        Args:
            context: Mapping of names to PyTensor variables.

        Returns:
            TensorVar: PyTensor expression representing the interpolated result.
        """
        # This is a complex implementation - for now return nominal
        # Full implementation would need the complete interpolation logic
        return context[self.nom]


class ProcessNormalizationFunction(Function):
    r"""
    Process normalization function with systematic variations.

    Implements the CMS Combine ProcessNormalization class which computes
    a normalization factor based on a nominal value and systematic variations.

    Parameters:
        name: Name of the function
        expression: Expression identifier (typically same as name)
        nominalValue: Base normalization value
        thetaList: Names of symmetric variation nuisance parameters
        logKappa: Symmetric log-normal variation factors
        asymmThetaList: Names of asymmetric variation nuisance parameters
        logAsymmKappa: Asymmetric [low, high] log-normal variation factors
        otherFactorList: Names of additional multiplicative factors
    """

    type: Literal["CMS::process_normalization"] = "CMS::process_normalization"
    expression_name: str = Field(alias="expression")
    nominalValue: float
    thetaList: list[str]
    logKappa: list[float]
    asymmThetaList: list[str]
    logAsymmKappa: list[list[float]]
    otherFactorList: list[str]

    @model_validator(mode="after")
    def process_parameters(self) -> ProcessNormalizationFunction:
        """Build the parameters dict from all parameter lists."""
        # All parameters this function depends on
        all_params = [*self.thetaList, *self.asymmThetaList, *self.otherFactorList]
        self.parameters = {name: name for name in all_params}
        return self

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ProcessNormalizationFunction:
        """Create a ProcessNormalizationFunction from dictionary configuration."""
        return cls(**config)

    def expression(self, _context: dict[str, TensorVar]) -> TensorVar:
        """
        Evaluate the process normalization function.

        Args:
            context: Mapping of names to PyTensor variables.

        Returns:
            TensorVar: PyTensor expression representing the normalization factor.
        """
        # This is a complex implementation - for now return nominal value as constant
        # Full implementation would need the complete normalization logic
        return cast(TensorVar, pt.constant(self.nominalValue))


# Define the union type for all function configurations
FunctionConfig = (
    SumFunction
    | ProductFunction
    | GenericFunction
    | InterpolationFunction
    | ProcessNormalizationFunction
)

registered_functions: dict[str, type[Function]] = {
    "sum": SumFunction,
    "product": ProductFunction,
    "generic_function": GenericFunction,
    "interpolation": InterpolationFunction,
    "CMS::process_normalization": ProcessNormalizationFunction,
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

    def __init__(self, funcs: list[dict[str, Any]]) -> None:
        """
        Collection of functions that compute parameter values.

        Args:
            funcs: List of function configurations from HS3 spec
        """
        self.funcs: dict[str, Function] = {}
        for func_config in funcs:
            func_type = func_config["type"]
            the_func = registered_functions.get(func_type)
            if the_func is None:
                msg = f"Unknown function type: {func_type}"
                raise ValueError(msg)
            func = the_func.from_dict(func_config)
            self.funcs[func.name] = func

    def __getitem__(self, item: str) -> Function:
        return self.funcs[item]

    def __contains__(self, item: str) -> bool:
        return item in self.funcs

    def __iter__(self) -> Iterator[Function]:
        return iter(self.funcs.values())

    def __len__(self) -> int:
        return len(self.funcs)
