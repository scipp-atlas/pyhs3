"""
HS3 Functions implementation.

Provides classes for handling HS3 functions including product functions,
generic functions with mathematical expressions, and interpolation functions.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pytensor.tensor as pt
from pydantic import (
    ConfigDict,
    Field,
    model_validator,
)
from pytensor.configdefaults import config

from pyhs3.base import balanced_sum
from pyhs3.context import Context
from pyhs3.distributions.histfactory.interpolations import (
    InterpolationDescriptor,
    apply_interpolation_descriptor,
    expand_interpolations,
)
from pyhs3.distributions.histogram import HistogramData
from pyhs3.functions.core import Function
from pyhs3.generic_parse import GenericExpressionMixin
from pyhs3.tensorutils import is_scalar_multiplicative_identity
from pyhs3.typing.aliases import TensorVar


def _asym_interpolation(
    theta: TensorVar, kappa_sum: float, kappa_diff: float
) -> TensorVar:
    """
    Implement asymmetric interpolation for ProcessNormalization.

    Based on the jaxfit implementation:
    https://github.com/nsmith-/jaxfit/blob/8479cd73e733ba35462287753fab44c0c560037b/src/jaxfit/roofit/combine.py#L197
    and CMS Combine's ``ProcessNormalization::logKappaForX`` logic.

    The shift is

    .. math::

        \\text{shift} = \\tfrac{1}{2}\\left(
            \\kappa_\\text{diff}\\,\\theta
            + \\kappa_\\text{sum}\\,\\theta\\,\\text{smoothStep}(\\theta)
        \\right)

    where ``smoothStep`` is the CMS smooth-step function

    .. math::

        \\text{smoothStep}(\\theta) = \\begin{cases}
            \\theta\\,(3\\theta^4 - 10\\theta^2 + 15)/8 & |\\theta| < 1 \\\\
            \\operatorname{sign}(\\theta) & |\\theta| \\geq 1
        \\end{cases}

    The factor multiplying ``kappa_sum`` is therefore
    :math:`\\theta\\cdot\\text{smoothStep}(\\theta)`, a smooth approximation of
    :math:`|\\theta|` that is ``0`` at :math:`\\theta=0`, equals ``1`` at
    :math:`|\\theta|=1`, and matches the first derivative of :math:`|\\theta|`
    at the boundary.

    Args:
        theta: The nuisance parameter value
        kappa_sum: logKappaHi + logKappaLo
        kappa_diff: logKappaHi - logKappaLo

    Returns:
        The interpolated shift value
    """
    # smoothStep(theta) for |theta| < 1: theta * (3*theta^4 - 10*theta^2 + 15) / 8
    # The factor multiplying kappa_sum is theta * smoothStep(theta), which for
    # |theta| < 1 equals theta^2 * (3*theta^4 - 10*theta^2 + 15) / 8 and for
    # |theta| >= 1 equals |theta| (a smooth approximation of |theta|).
    theta_sq = theta * theta
    theta_quad = theta_sq * theta_sq
    poly_result = theta_sq * (3.0 * theta_quad - 10.0 * theta_sq + 15.0) / 8.0

    # Linear behaviour for |theta| >= 1: theta * sign(theta) == |theta|
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
    summands: list[int | float | str] = Field(..., repr=False)

    def _expression(self, context: Context) -> TensorVar:
        """
        Evaluate the sum function.

        Args:
            context: Mapping of names to PyTensor variables.

        Returns:
            TensorVar: PyTensor expression representing the sum of all summands.
        """
        terms = self.get_parameter_list(context, "summands")
        return balanced_sum(terms, pt.constant(0.0))


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
            if (
                is_scalar_multiplicative_identity(factor_value)
                and factor_value.dtype == result.dtype
            ):
                continue

            if (
                is_scalar_multiplicative_identity(result)
                and result.dtype == factor_value.dtype
            ):
                result = factor_value
                continue

            result = result * factor_value

        return result


class GenericFunction(GenericExpressionMixin, Function):
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
        :external+hs3:ref:`generic <hs3.generic-function>`
    """

    type: Literal["generic", "generic_function"] = Field(default="generic", repr=False)

    @model_validator(mode="before")
    @classmethod
    def _canonicalize_legacy_type(cls, data: object) -> object:
        """Accept pyhs3's historical tag but always retain ROOT's canonical one."""
        if isinstance(data, dict) and data.get("type") == "generic_function":
            data = {**data, "type": "generic"}
        return data

    def _expression(self, context: Context) -> TensorVar:
        """
        Evaluate the generic function expression.

        Args:
            context: Mapping of names to PyTensor variables.

        Returns:
            TensorVar: PyTensor expression representing the parsed mathematical expression.
        """
        return self._eval_expression(context)


class InterpolationFunction(Function):
    """ROOT ``PiecewiseInterpolation`` with structured interpolation forms."""

    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    type: Literal["interpolation"] = Field(default="interpolation", repr=False)
    high: list[str] = Field(..., repr=False)
    low: list[str] = Field(..., repr=False)
    nom: str = Field(..., repr=False)
    interpolations: list[InterpolationDescriptor] = Field(
        ...,
        repr=False,
        json_schema_extra={"preprocess": False},
    )
    positiveDefinite: bool = Field(..., repr=False)
    vars: list[str] = Field(..., repr=False)

    @model_validator(mode="after")
    def _validate_interpolation_lengths(self) -> InterpolationFunction:
        n_parameters = len(self.vars)
        if len(self.low) != n_parameters or len(self.high) != n_parameters:
            msg = (
                f"PiecewiseInterpolation '{self.name}' has non-matching lengths "
                "of 'vars', 'high' and 'low'"
            )
            raise ValueError(msg)
        expanded = expand_interpolations(
            self.interpolations,
            n_parameters,
            "piecewise",
        )
        if expanded and all(item == expanded[0] for item in expanded[1:]):
            self.interpolations = [expanded[0]]
        else:
            self.interpolations = expanded
        return self

    def _expression(self, context: Context) -> TensorVar:
        nominal = context[self._parameters["nom"]]
        result = nominal
        descriptors = expand_interpolations(
            self.interpolations,
            len(self.vars),
            "piecewise",
        )
        variables = self.get_parameter_list(context, "vars")
        highs = self.get_parameter_list(context, "high")
        lows = self.get_parameter_list(context, "low")
        for descriptor, alpha, high, low in zip(
            descriptors, variables, highs, lows, strict=True
        ):
            result = apply_interpolation_descriptor(
                descriptor,
                alpha,
                nominal,
                high,
                low,
                current=result,
            )

        if self.positiveDefinite:
            zero = pt.constant(0.0, dtype=result.dtype)
            result = pt.maximum(result, zero)

        return result


class Interpolation0DFunction(Function):
    """ROOT ``FlexibleInterpVar`` with numeric nominal and anchor values."""

    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    type: Literal["interpolation0d"] = Field(default="interpolation0d", repr=False)
    high: list[float] = Field(
        ...,
        repr=False,
        json_schema_extra={"preprocess": False},
    )
    low: list[float] = Field(
        ...,
        repr=False,
        json_schema_extra={"preprocess": False},
    )
    nom: float = Field(
        ...,
        repr=False,
        json_schema_extra={"preprocess": False},
    )
    interpolations: list[InterpolationDescriptor] = Field(
        ...,
        repr=False,
        json_schema_extra={"preprocess": False},
    )
    vars: list[str] = Field(..., repr=False)

    @model_validator(mode="after")
    def _validate_interpolation_lengths(self) -> Interpolation0DFunction:
        n_parameters = len(self.vars)
        if len(self.low) != n_parameters or len(self.high) != n_parameters:
            msg = (
                f"FlexibleInterpVar '{self.name}' has non-matching lengths of "
                "'vars', 'high' and 'low'"
            )
            raise ValueError(msg)
        expanded = expand_interpolations(
            self.interpolations,
            n_parameters,
            "flexible",
        )
        if expanded and all(item == expanded[0] for item in expanded[1:]):
            self.interpolations = [expanded[0]]
        else:
            self.interpolations = expanded
        return self

    def _expression(self, context: Context) -> TensorVar:
        # RooFit stores the numeric FlexibleInterpVar payload as doubles.
        nominal = cast(
            TensorVar,
            pt.constant(np.asarray(self.nom, dtype=config.floatX)),
        )
        result = nominal
        descriptors = expand_interpolations(
            self.interpolations,
            len(self.vars),
            "flexible",
        )
        variables = self.get_parameter_list(context, "vars")
        for descriptor, alpha, high_value, low_value in zip(
            descriptors,
            variables,
            self.high,
            self.low,
            strict=True,
        ):
            high = cast(TensorVar, pt.constant(high_value, dtype=nominal.dtype))
            low = cast(TensorVar, pt.constant(low_value, dtype=nominal.dtype))
            result = apply_interpolation_descriptor(
                descriptor,
                alpha,
                nominal,
                high,
                low,
                current=result,
            )

        tiny_value = np.finfo(np.dtype(result.dtype)).tiny
        tiny = pt.constant(tiny_value, dtype=result.dtype)
        zero = pt.constant(0.0, dtype=result.dtype)
        result = pt.where(  # type: ignore[no-untyped-call]
            result <= zero, tiny, result
        )

        return cast(TensorVar, result)


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
        sym_terms = [
            (self.logKappa[i] if i < len(self.logKappa) else 0.0) * context[theta_name]
            for i, theta_name in enumerate(self.thetaList)
        ]
        symShift = balanced_sum(sym_terms, pt.constant(0.0))

        # Asymmetric variations: use asymmetric interpolation
        asym_terms = []
        for i, theta_name in enumerate(self.asymmThetaList):
            theta = context[theta_name]
            log_kappa_lo, log_kappa_hi = self.logAsymmKappa[i]
            kappa_sum = log_kappa_hi + log_kappa_lo
            kappa_diff = log_kappa_hi - log_kappa_lo
            asym_terms.append(_asym_interpolation(theta, kappa_sum, kappa_diff))
        asymShift = balanced_sum(asym_terms, pt.constant(0.0))

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

    Implements ROOT's ``RooRecursiveFraction`` which computes a recursive
    fraction. Used for constructing a set of fractions that automatically sum
    to one (e.g. ``RooAddPdf`` with ``recursiveFractions=True``).

    For a coefficient list :math:`(a_0, a_1, \dots, a_{n-1})` ROOT's
    ``RooRecursiveFraction::evaluate()`` returns

    .. math::

        f = a_0 \prod_{i=1}^{n-1} (1 - a_i)

    so that the leading coefficient is scaled by the complement of all the
    remaining ones. A single coefficient returns :math:`a_0` itself
    (empty product).

    Example:
        :math:`(0.2, 0.5, 0.5) \to 0.2 \cdot (1-0.5) \cdot (1-0.5) = 0.05`.

    The non-recursive branch keeps the simple normalization
    :math:`a_0 / \sum_j a_j` for the (rare) flat-fraction convention.

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
            # Simple normalization: a_0 / sum(all)
            total = sum(coeffs)
            return cast(TensorVar, coeffs[0] / total)

        # Recursive fraction (ROOT RooRecursiveFraction::evaluate):
        #   f = a_0 * prod_{i>=1} (1 - a_i)
        # A single coefficient yields a_0 (empty product).
        result: TensorVar = coeffs[0]
        for coeff in coeffs[1:]:
            result = cast(TensorVar, result * (1.0 - coeff))

        return result


# Registry for functions defined in this module
# NOTE: HistogramFunction is intentionally NOT registered here because it has no
# _expression() implementation. Workspaces referencing "histogram" will get
# the normal clean unknown-type validation error from the discriminated union.
functions: dict[str, type[Function]] = {
    "sum": SumFunction,
    "product": ProductFunction,
    "generic": GenericFunction,
    "generic_function": GenericFunction,
    "interpolation": InterpolationFunction,
    "interpolation0d": Interpolation0DFunction,
    "CMS::process_normalization": ProcessNormalizationFunction,
    "CMS::asympow": CMSAsymPowFunction,
    "roorecursivefraction_dist": RooRecursiveFractionFunction,
}
