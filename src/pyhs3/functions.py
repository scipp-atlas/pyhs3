"""
HS3 Functions implementation.

Provides classes for handling HS3 functions including product functions,
generic functions with mathematical expressions, and interpolation functions.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Annotated, Any, Literal, TypeVar, cast

import pytensor.tensor as pt
import sympy as sp
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    RootModel,
    model_validator,
)

from pyhs3.exceptions import UnknownInterpolationCodeError, custom_error_msg
from pyhs3.generic_parse import analyze_sympy_expr, parse_expression, sympy_to_pytensor
from pyhs3.typing.aliases import TensorVar

log = logging.getLogger(__name__)


FuncT = TypeVar("FuncT", bound="Function")


class Function(BaseModel):
    """Base class for HS3 functions."""

    model_config = ConfigDict(serialize_by_alias=True)

    name: str
    type: str
    _parameters: dict[str, str] = PrivateAttr(default_factory=dict)
    _constants_values: dict[str, float | int] = PrivateAttr(default_factory=dict)

    @property
    def parameters(self) -> dict[str, str]:
        """Access to parameter mapping."""
        return self._parameters

    @property
    def constants(self) -> dict[str, TensorVar]:
        """Convert stored numeric constants to PyTensor constants."""
        return {
            name: cast(TensorVar, pt.constant(value))
            for name, value in self._constants_values.items()
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
        self._parameters = {name: name for name in self.summands}
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
        self._parameters = {name: name for name in self.factors}
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
    _sympy_expr: sp.Expr = PrivateAttr(default=None)
    _dependent_vars: list[str] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def setup_expression(self) -> GenericFunction:
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
        variables = [context[name] for name in self._parameters.values()]

        # Convert using the pre-parsed sympy expression
        return sympy_to_pytensor(self._sympy_expr, variables)


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
        self._parameters = {name: name for name in all_params}
        return self

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> InterpolationFunction:
        """Create an InterpolationFunction from dictionary configuration."""
        return cls(**config)

    def _flexible_interp_single(
        self,
        interp_code: int,
        low_val: TensorVar,
        high_val: TensorVar,
        boundary: float,
        nominal: TensorVar,
        param_val: TensorVar,
    ) -> TensorVar:
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
                TensorVar,
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
                TensorVar,
                pt.switch(
                    param_val >= 0,
                    cast(TensorVar, pt.power(ratio_high, param_val)) - 1.0,  # type: ignore[no-untyped-call]
                    cast(TensorVar, pt.power(ratio_low, -param_val)) - 1.0,  # type: ignore[no-untyped-call]
                ),
            )

        if interp_code == 2:
            # Exponential interpolation, linear extrapolation (additive)
            return cast(
                TensorVar,
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
                TensorVar,
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
                TensorVar,
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
                TensorVar,
                pt.switch(
                    pt.abs(param_val) >= boundary,
                    # Exponential extrapolation for |theta| >= 1
                    pt.switch(
                        param_val >= 0,
                        cast(TensorVar, pt.power(ratio_high, param_val)) - 1.0,  # type: ignore[no-untyped-call]
                        cast(TensorVar, pt.power(ratio_low, -param_val)) - 1.0,  # type: ignore[no-untyped-call]
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
            TensorVar,
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

    def expression(self, context: dict[str, TensorVar]) -> TensorVar:
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
                - Nuisance parameters (referenced by `vars` list)

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
        for i, var_name in enumerate(self.vars):
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
        self._parameters = {name: name for name in all_params}
        return self

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ProcessNormalizationFunction:
        """Create a ProcessNormalizationFunction from dictionary configuration."""
        return cls(**config)

    def _asym_interpolation(
        self, theta: TensorVar, kappa_sum: float, kappa_diff: float
    ) -> TensorVar:
        """
        Implement asymmetric interpolation function.

        Based on CMS Combine's _asym_interpolation function.

        Args:
            theta: Nuisance parameter value
            kappa_sum: Sum of low and high kappa values
            kappa_diff: Difference of high and low kappa values

        Returns:
            Interpolated value
        """
        abs_theta = pt.abs(theta)

        # Polynomial coefficients for smooth interpolation
        # Based on _asym_poly = jnp.array([3.0, -10.0, 15.0, 0.0]) / 8.0
        poly_result = (3.0 * theta**4 - 10.0 * theta**2 + 15.0) / 8.0

        # Choose between linear extrapolation (|theta| > 1) and polynomial interpolation (|theta| <= 1)
        smooth_function = pt.switch(abs_theta > 1.0, abs_theta, poly_result)

        # Apply asymmetric interpolation formula
        morph = 0.5 * (kappa_diff * theta + kappa_sum * smooth_function)

        return cast(TensorVar, morph)

    def expression(self, context: dict[str, TensorVar]) -> TensorVar:
        """
        Evaluate the process normalization function.

        Args:
            context: Mapping of names to PyTensor variables.

        Returns:
            TensorVar: PyTensor expression representing the normalization factor.
        """
        # Start with the nominal value
        result = pt.constant(self.nominalValue)

        # Add symmetric variations
        sym_shift = pt.constant(0.0)
        for theta_name, kappa in zip(self.thetaList, self.logKappa, strict=False):
            theta = context[theta_name]
            sym_shift = sym_shift + kappa * theta

        # Add asymmetric variations
        asym_shift = pt.constant(0.0)
        for theta_name, kappa_pair in zip(
            self.asymmThetaList, self.logAsymmKappa, strict=False
        ):
            theta = context[theta_name]
            kappa_lo, kappa_hi = kappa_pair
            kappa_sum = kappa_hi + kappa_lo
            kappa_diff = kappa_hi - kappa_lo
            asym_contribution = self._asym_interpolation(theta, kappa_sum, kappa_diff)
            asym_shift = asym_shift + asym_contribution

        # Apply exponential of total shift
        result = result * pt.exp(sym_shift + asym_shift)

        # Multiply by additional factors
        for factor_name in self.otherFactorList:
            factor = context[factor_name]
            result = result * factor

        return cast(TensorVar, result)


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

# Type alias for all function types using discriminated union
FunctionType = Annotated[
    SumFunction
    | ProductFunction
    | GenericFunction
    | InterpolationFunction
    | ProcessNormalizationFunction,
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
