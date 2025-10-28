"""
HS3 Functions implementation.

Provides classes for handling HS3 functions including product functions,
generic functions with mathematical expressions, and interpolation functions.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import Annotated, Literal, cast

import pytensor.tensor as pt
import sympy as sp
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
)

from pyhs3.context import Context
from pyhs3.data import Axis
from pyhs3.exceptions import custom_error_msg
from pyhs3.functions.core import Function
from pyhs3.generic_parse import analyze_sympy_expr, parse_expression, sympy_to_pytensor
from pyhs3.typing.aliases import TensorVar

log = logging.getLogger(__name__)


def _asym_interpolation(
    theta: TensorVar, kappa_sum: float, kappa_diff: float
) -> TensorVar:
    """
    Implement asymmetric interpolation for ProcessNormalization.

    Based on the jaxfit implementation:
    https://github.com/nsmith-/jaxfit/blob/8479cd73e733ba35462287753fab44c0c560037b/src/jaxfit/roofit/combine.py#L197
    and CMS Combine logic.
    Uses polynomial interpolation for |theta| < 1 and linear extrapolation beyond.

    Args:
        theta: The nuisance parameter value
        kappa_sum: logKappaHi + logKappaLo
        kappa_diff: logKappaHi - logKappaLo

    Returns:
        The interpolated shift value
    """
    # Polynomial interpolation for |theta| < 1
    # Polynomial: (3*theta^4 - 10*theta^2 + 15) / 8
    theta_sq = theta * theta
    theta_quad = theta_sq * theta_sq
    poly_result = (3.0 * theta_quad - 10.0 * theta_sq + 15.0) / 8.0

    # Linear extrapolation for |theta| >= 1
    linear_result = pt.abs(theta)

    # Choose between polynomial and linear based on |theta|
    abs_theta = pt.abs(theta)
    smooth_function = cast(
        TensorVar, pt.switch(abs_theta < 1.0, poly_result, linear_result)
    )

    # Final asymmetric interpolation formula
    return cast(TensorVar, 0.5 * (kappa_diff * theta + kappa_sum * smooth_function))


class SumFunction(Function):
    """Sum function that adds summands together.

    HS3 Reference:
        :ref:`hs3:hs3.sum`
    """

    type: Literal["sum"] = Field(default="sum", repr=False)
    summands: list[str] = Field(..., repr=False)

    def _expression(self, context: Context) -> TensorVar:
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
    """Product function that multiplies factors together.

    HS3 Reference:
        :ref:`hs3:hs3.product`
    """

    type: Literal["product"] = Field(default="product", repr=False)
    factors: list[int | float | str] = Field(..., repr=False)

    def _expression(self, context: Context) -> TensorVar:
        """
        Evaluate the product function.

        Args:
            context: Mapping of names to PyTensor variables.

        Returns:
            TensorVar: PyTensor expression representing the product of all factors.
        """
        if not self.factors:
            return pt.constant(1.0)

        # Get list of factors using flattened parameter keys
        factor_values = self.get_parameter_list(context, "factors")
        result = factor_values[0]
        for factor_value in factor_values[1:]:
            result = result * factor_value

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

    HS3 Reference:
        :hs3:label:`generic_function <hs3.generic-function>`
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, serialize_by_alias=True)

    type: Literal["generic_function"] = Field(default="generic_function", repr=False)
    expression_str: str = Field(alias="expression", repr=False)
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

    def _expression(self, context: Context) -> TensorVar:
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


class InterpolationCode(IntEnum):
    """
    Enumeration of interpolation codes for systematic variations.

    Defines the different interpolation methods used by InterpolationFunction
    for systematic uncertainty variations. Each code represents a different
    mathematical approach to interpolating between nominal, low, and high values.
    """

    LIN_LIN_ADD = 0
    EXP_EXP_MUL = 1
    EXP_LIN_ADD = 2
    EXP_MIX_ADD = 3
    POL_LIN_ADD = 4
    POL_EXP_MUL = 5
    POL_LIN_MUL = 6


class InterpolationFunction(Function):
    r"""
    Piecewise interpolation function implementation.

    Implements ROOT's PiecewiseInterpolation logic to morph between nominal
    and variation distributions based on nuisance parameter values.
    Supports multiple interpolation codes (0-6) for different mathematical approaches.

    HS3 Reference:
        Note: Interpolation functions are not explicitly defined in the current HS3 specification.

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

    model_config = ConfigDict(use_enum_values=True)

    type: Literal["interpolation"] = Field(default="interpolation", repr=False)
    high: list[str] = Field(..., repr=False)
    low: list[str] = Field(..., repr=False)
    nom: str = Field(..., repr=False)
    interpolationCodes: Annotated[
        list[InterpolationCode],
        custom_error_msg(
            {
                "enum": "Unknown interpolation code {input} in function '{name}'. Valid codes are {expected}."
            }
        ),
    ] = Field(..., repr=False)
    positiveDefinite: bool = Field(..., repr=False)
    vars: list[str] = Field(..., repr=False)

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

    def _expression(self, context: Context) -> TensorVar:
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
    a normalization factor based on systematic variations. This matches
    the actual CMS Combine implementation and JSON structure from combine files.

    Mathematical formulation:
        result = nominalValue * exp(symShift + asymShift) * otherFactors

        where:
        - symShift = sum(logKappa[i] * theta[i]) for symmetric variations
        - asymShift = sum(_asym_interpolation(theta[i], kappa_sum[i], kappa_diff[i]))
          for asymmetric variations with kappa_sum = logKappaHi + logKappaLo
          and kappa_diff = logKappaHi - logKappaLo
        - otherFactors = product of all additional multiplicative terms

    Parameters:
        name: Name of the function
        nominalValue: Baseline normalization value (default 1.0)
        thetaList: Names of symmetric variation nuisance parameters
        logKappa: Log-kappa values for symmetric variations (optional, defaults to empty)
        asymmThetaList: Names of asymmetric variation nuisance parameters
        logAsymmKappa: List of [logKappaLo, logKappaHi] pairs for asymmetric variations (optional)
        otherFactorList: Names of additional multiplicative factors
    """

    type: Literal["CMS::process_normalization"] = Field(
        default="CMS::process_normalization", repr=False
    )
    nominalValue: float = Field(
        default=1.0, json_schema_extra={"preprocess": False}, repr=False
    )
    thetaList: list[str] = Field(default_factory=list, repr=False)
    logKappa: list[float] = Field(
        default_factory=list, json_schema_extra={"preprocess": False}, repr=False
    )
    asymmThetaList: list[str] = Field(default_factory=list, repr=False)
    logAsymmKappa: list[list[float]] = Field(
        default_factory=list, json_schema_extra={"preprocess": False}, repr=False
    )
    otherFactorList: list[str] = Field(default_factory=list, repr=False)

    def _expression(self, context: Context) -> TensorVar:
        """
        Evaluate the process normalization function.

        Implements the full CMS Combine ProcessNormalization logic:
        result = nominalValue * exp(symShift + asymShift) * otherFactors

        Args:
            context: Mapping of names to PyTensor variables.

        Returns:
            TensorVar: PyTensor expression representing the normalization factor.
        """
        # Start with nominal value
        result = pt.constant(self.nominalValue)

        # Symmetric variations: symShift = sum(logKappa[i] * theta[i])
        symShift = pt.constant(0.0)
        for i, theta_name in enumerate(self.thetaList):
            theta = context[theta_name]
            # Use provided logKappa value if available, otherwise assume 0.0 (no effect)
            log_kappa = self.logKappa[i] if i < len(self.logKappa) else 0.0
            symShift = symShift + log_kappa * theta

        # Asymmetric variations: use asymmetric interpolation
        asymShift = pt.constant(0.0)
        for i, theta_name in enumerate(self.asymmThetaList):
            theta = context[theta_name]
            log_kappa_lo, log_kappa_hi = self.logAsymmKappa[i]
            kappa_sum = log_kappa_hi + log_kappa_lo
            kappa_diff = log_kappa_hi - log_kappa_lo
            asymShift = asymShift + _asym_interpolation(theta, kappa_sum, kappa_diff)

        # Apply exponential scaling: nominal * exp(symShift + asymShift)
        result = result * pt.exp(symShift + asymShift)

        # Multiply by additional factors
        for factor_name in self.otherFactorList:
            factor = context[factor_name]
            result = result * factor

        return cast(TensorVar, result)


class CMSAsymPowFunction(Function):
    r"""
    CMS AsymPow function implementation.

    Implements CMS's AsymPow function which provides asymmetric power-law
    variations for systematic uncertainties. Used in CMS combine for
    asymmetric systematic variations.

    .. math::

        f(\theta; \kappa_{low}, \kappa_{high}) = \begin{cases}
        \kappa_{low}^{-\theta}, & \text{if } \theta < 0 \\
        \kappa_{high}^{\theta}, & \text{if } \theta \geq 0
        \end{cases}

    Parameters:
        name: Name of the function
        kappaLow: Low-side variation factor (used for θ < 0)
        kappaHigh: High-side variation factor (used for θ ≥ 0)
        theta: Parameter name for the nuisance parameter
    """

    type: Literal["CMS::asympow"] = Field(default="CMS::asympow", repr=False)
    kappaLow: str | float | int = Field(..., repr=False)
    kappaHigh: str | float | int = Field(..., repr=False)
    theta: str = Field(..., repr=False)

    def _expression(self, context: Context) -> TensorVar:
        """
        Evaluate the AsymPow function.

        Args:
            context: Mapping of names to PyTensor variables.

        Returns:
            TensorVar: PyTensor expression representing the asymmetric power function.
        """
        kappa_low = context[self._parameters["kappaLow"]]
        kappa_high = context[self._parameters["kappaHigh"]]
        theta = context[self._parameters["theta"]]

        # AsymPow: kappaLow^(-theta) for theta < 0, kappaHigh^theta for theta >= 0
        return cast(
            TensorVar,
            pt.switch(
                theta < 0,
                cast(TensorVar, pt.power(kappa_low, -theta)),  # type: ignore[no-untyped-call]
                cast(TensorVar, pt.power(kappa_high, theta)),  # type: ignore[no-untyped-call]
            ),
        )


class HistogramData(BaseModel):
    """
    Histogram data implementation for the HistogramFunction.

    Parameters:
        axes: list of Axis used to describe the binning
        contents: list of bin content parameter values
    """

    model_config = ConfigDict()

    axes: list[Axis] = Field(..., repr=False)
    contents: list[float] = Field(..., repr=False)


class HistogramFunction(Function):
    r"""
    Histogram function implementation.

    Implements a histogram-based function that provides piecewise constant
    values based on bin lookup. Used for non-parametric functions and
    data-driven backgrounds.

    .. math::

        f(x) = h_i \quad \text{where } x \in \text{bin}_i

    Parameters:
        name: Name of the function
        data: histogram data with binning and contents
    """

    type: Literal["histogram"] = Field(default="histogram", repr=False)
    data: HistogramData = Field(
        ..., json_schema_extra={"preprocess": False}, repr=False
    )


class RooRecursiveFractionFunction(Function):
    r"""
    ROOT RooRecursiveFraction function implementation.

    Implements ROOT's RooRecursiveFraction which computes fractions recursively.
    Used for constrained fraction calculations where fractions must sum to 1.

    .. math::

        f_i = \frac{a_i}{\sum_{j=i}^n a_j}

    where the recursive fractions ensure proper normalization.

    Parameters:
        name: Name of the function
        coefficients: List of coefficient parameter names
        recursive: Whether to use recursive fraction calculation
    """

    type: Literal["roorecursivefraction_dist"] = Field(
        default="roorecursivefraction_dist", repr=False
    )
    coefficients: list[int | float | str] = Field(alias="list", repr=False)
    recursive: bool = Field(default=True, repr=False)

    def _expression(self, context: Context) -> TensorVar:
        """
        Evaluate the recursive fraction function.

        Args:
            context: Mapping of names to PyTensor variables.

        Returns:
            TensorVar: PyTensor expression representing the recursive fraction.
        """
        if not self.coefficients:
            return cast(TensorVar, pt.constant(0.0))

        coeffs = self.get_parameter_list(context, "coefficients")

        if not self.recursive:
            # Simple normalization
            total = sum(coeffs)
            return cast(TensorVar, coeffs[0] / total)

        # Recursive fraction calculation
        # For first coefficient: a_0 / (a_0 + a_1 + ... + a_n)
        # For i-th coefficient: a_i / (a_i + a_{i+1} + ... + a_n) * (1 - sum of previous fractions)

        if len(coeffs) == 1:
            return cast(TensorVar, pt.constant(1.0))

        # Calculate the first recursive fraction: a_0 / sum(all)
        total_sum = sum(coeffs)
        first_fraction = coeffs[0] / total_sum

        return cast(TensorVar, first_fraction)


# Registry for functions defined in this module
functions: dict[str, type[Function]] = {
    "sum": SumFunction,
    "product": ProductFunction,
    "generic_function": GenericFunction,
    "interpolation": InterpolationFunction,
    "CMS::process_normalization": ProcessNormalizationFunction,
    "CMS::asympow": CMSAsymPowFunction,
    "histogram": HistogramFunction,  # type: ignore[type-abstract]
    "roorecursivefraction_dist": RooRecursiveFractionFunction,
}
