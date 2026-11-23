"""
HS3 Functions implementation.

Provides classes for handling HS3 functions including product functions,
generic functions with mathematical expressions, and interpolation functions.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Annotated, Literal, cast

import numpy as np
import numpy.typing as npt
import pytensor.tensor as pt
from pydantic import (
    ConfigDict,
    Field,
)
from pytensor.tensor.shape import specify_shape
from pytensor.tensor.variable import TensorConstant

from pyhs3.base import balanced_sum
from pyhs3.context import Context
from pyhs3.distributions.histfactory.interpolations import (
    interpolate_code0,
    interpolate_code1,
    interpolate_code4,
    interpolate_parabolic,
    interpolate_poly6,
)
from pyhs3.distributions.histogram import HistogramData
from pyhs3.exceptions import custom_error_msg
from pyhs3.functions.core import BatchableFunction, Function
from pyhs3.generic_parse import GenericExpressionMixin
from pyhs3.tensorutils import is_scalar_multiplicative_identity
from pyhs3.typing.aliases import TensorVar

log = logging.getLogger(__name__)


def _pad_to_ndim(var: TensorVar, ndim: int) -> TensorVar:
    """Prepend broadcastable axes to ``var`` until it has ``ndim`` dimensions."""
    if var.ndim == ndim:
        return var
    pattern = ("x",) * (ndim - var.ndim) + tuple(range(var.ndim))
    return cast(TensorVar, var.dimshuffle(pattern))  # type: ignore[no-untyped-call]


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
    summands: list[str] = Field(..., repr=False)

    def _expression(self, context: Context) -> TensorVar:
        """
        Evaluate the sum function.

        Args:
            context: Mapping of names to PyTensor variables.

        Returns:
            TensorVar: PyTensor expression representing the sum of all summands.
        """
        terms = [context[summand] for summand in self.summands]
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
        :external+hs3:ref:`generic_function <hs3.generic-function>`
    """

    type: Literal["generic_function"] = Field(default="generic_function", repr=False)

    def _expression(self, context: Context) -> TensorVar:
        """
        Evaluate the generic function expression.

        Args:
            context: Mapping of names to PyTensor variables.

        Returns:
            TensorVar: PyTensor expression representing the parsed mathematical expression.
        """
        return self._eval_expression(context)


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


class InterpolationFunction(BatchableFunction):
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

    @staticmethod
    def _flexible_interp_single(
        interp_code: int,
        low_val: TensorVar,
        high_val: TensorVar,
        boundary: float,
        nominal: TensorVar,
        param_val: TensorVar,
    ) -> TensorVar:
        r"""
        Implement flexible interpolation for a single parameter.

        Based on ROOT's ``FlexibleInterpVar`` /
        ``RooFit::Detail::MathFuncs::flexibleInterpSingle`` with support for
        interpolation codes 0-6. This method computes the interpolation
        contribution :math:`I_i(\theta_i)` for a single nuisance parameter.

        The verified-correct building blocks from
        :mod:`pyhs3.distributions.histfactory.interpolations` are reused
        directly (those return the full ``nominal + delta`` value, so the
        nominal/baseline is subtracted off here to recover the additive delta
        or multiplicative factor that ROOT's ``flexibleInterpSingle`` returns):

        - **Code 0** (additive): piecewise-linear, ``interpolate_code0``.
        - **Code 1** (multiplicative): piecewise-exponential, ``interpolate_code1``.
        - **Codes 2, 3** (additive): parabolic interpolation with linear
          extrapolation, ``interpolate_parabolic``. ROOT maps code 3 onto code 2.
        - **Code 4** (additive): 6th-degree polynomial interpolation with
          linear extrapolation, ``interpolate_poly6``.
        - **Code 5** (multiplicative): 6th-degree polynomial interpolation in
          log space with exponential extrapolation, ``interpolate_code4``.
        - **Code 6** (multiplicative): 6th-degree polynomial interpolation with
          linear extrapolation in ratio space, ``interpolate_poly6`` applied to
          ``high/nominal`` and ``low/nominal``.

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
            - Codes 1,5,6: Multiplicative factor (already with 1 subtracted)
        """
        # The interpolations helpers assume a boundary of 1.0; ProcessNormalization
        # / PiecewiseInterpolation always use this boundary.
        del boundary

        if interp_code == 0:
            # Piecewise-linear interpolation/extrapolation (additive)
            return cast(
                TensorVar,
                interpolate_code0(param_val, nominal, high_val, low_val) - nominal,
            )

        if interp_code == 1:
            # Piecewise-exponential interpolation/extrapolation (multiplicative)
            return cast(
                TensorVar,
                interpolate_code1(param_val, nominal, high_val, low_val) / nominal
                - 1.0,
            )

        if interp_code in (2, 3):
            # Parabolic interpolation with linear extrapolation (additive).
            # ROOT converts code 3 to code 2. interpolate_parabolic matches ROOT's
            # a*alpha^2 + b*alpha central region with continuous linear extensions.
            return cast(
                TensorVar,
                interpolate_parabolic(param_val, nominal, high_val, low_val) - nominal,
            )

        if interp_code == 4:
            # 6th-degree polynomial interpolation + linear extrapolation (additive)
            return cast(
                TensorVar,
                interpolate_poly6(param_val, nominal, high_val, low_val) - nominal,
            )

        if interp_code == 5:
            # 6th-degree polynomial in log space + exponential extrapolation
            # (multiplicative)
            return cast(
                TensorVar,
                interpolate_code4(param_val, nominal, high_val, low_val) / nominal
                - 1.0,
            )

        # Code 6: 6th-degree polynomial + linear extrapolation in ratio space
        # (multiplicative). Work in nominal=1 ratio space, as ROOT does.
        one = pt.constant(1.0)
        ratio_high = high_val / nominal
        ratio_low = low_val / nominal
        return cast(
            TensorVar,
            interpolate_poly6(param_val, one, ratio_high, ratio_low) - 1.0,
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

        for additive, run in self._entry_runs():
            by_code: dict[int, list[int]] = {}
            for i in run:
                by_code.setdefault(self.interpolationCodes[i], []).append(i)
            terms = [
                (
                    len(indices),
                    self._stacked_contribution(code, indices, nominal, context),
                )
                for code, indices in by_code.items()
            ]
            if additive:
                # Each term has shape (K_code, *batch); summing over the entry
                # axis gives shape (*batch).
                result = balanced_sum(
                    [result, *(pt.sum(term, axis=0) for _, term in terms)],  # type: ignore[no-untyped-call]
                    result,
                )
            else:
                one = pt.constant(1.0, dtype=terms[0][1].dtype)
                # Product over the entry axis of each (K_code, *batch) factor
                # tensor, giving shape (*batch).  no_zeros_in_input=True makes
                # the gradient the elementwise prod / factor instead of going
                # through ProdWithoutZeros (no JAX dispatch, no second
                # derivative).  Splitting the factors into one row per entry
                # would avoid that too, but autodiff then scatters every row's
                # gradient back with an IncSubtensor, which with thousands of
                # entries makes the gradient graph very slow to compile.  A
                # factor of exactly zero (a multiplicative yield of zero) gives
                # a non-finite gradient.
                # One n-ary Mul per run: the nominal (or running total) times
                # every code group's product.  The Python loop only builds a
                # fixed graph over the few code groups; each pt.prod reduces
                # all of its group's entries in one vectorised op.
                result = pt.mul(
                    result,
                    *(
                        pt.prod(one + term, axis=0, no_zeros_in_input=True)  # type: ignore[no-untyped-call]
                        for _, term in terms
                    ),
                )

        # Apply positive definite constraint if requested
        if self.positiveDefinite:
            result = pt.maximum(result, 0.0)

        return result

    def _entry_runs(self) -> list[tuple[bool, list[int]]]:
        """Group entry indices into consecutive runs of the same kind.

        Entries are applied in order (additive codes add to the running total,
        multiplicative codes scale it), so interleaved additive and
        multiplicative entries do not commute.  Within a consecutive run of the
        same kind they do, so each run is evaluated with one vectorised
        expression per interpolation code instead of one expression per entry.
        This keeps the graph size independent of the number of entries.

        Entries whose index exceeds the ``high``/``low``/``interpolationCodes``
        lists are skipped with a warning.

        Returns:
            ``(additive, indices)`` pairs in entry order.
        """
        runs: list[tuple[bool, list[int]]] = []
        for i in range(len(self.vars)):
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
            additive = self.interpolationCodes[i] in (0, 2, 3, 4)
            if runs and runs[-1][0] == additive:
                runs[-1][1].append(i)
            else:
                runs.append((additive, [i]))
        return runs

    def _stacked_contribution(
        self,
        code: int,
        indices: list[int],
        nominal: TensorVar,
        context: Context,
    ) -> TensorVar:
        """Evaluate the contributions of several same-code entries at once.

        Stacks the entries' parameters and variations along a new leading axis
        and evaluates the elementwise interpolation formula once.

        Args:
            code: Interpolation code shared by all ``indices``.
            indices: Entry positions in ``vars``/``high``/``low``.
            nominal: Nominal value, shape ``(*batch)`` or scalar.
            context: Mapping of names to PyTensor variables.

        Returns:
            Contributions of shape ``(K, *batch)`` with ``K = len(indices)``;
            row ``k`` is :meth:`_flexible_interp_single` for ``indices[k]``.
        """
        columns = [
            [context[names[i]] for i in indices]
            for names in (self.vars, self.low, self.high)
        ]
        # Entries may mix scalars and batched (M,) vectors; give them a common
        # shape before stacking so the leading axis indexes entries only.
        ndim = max(nominal.ndim, *(var.ndim for column in columns for var in column))
        padded = [[_pad_to_ndim(var, ndim) for var in column] for column in columns]
        flat = [var for column in padded for var in column]
        if len({var.type for var in flat}) > 1:
            flat = [cast(TensorVar, var) for var in pt.broadcast_arrays(*flat)]
        k = len(indices)
        params, lows, highs = (pt.stack(flat[j * k : (j + 1) * k]) for j in range(3))
        return self._flexible_interp_single(
            interp_code=code,
            low_val=lows,
            high_val=highs,
            boundary=1.0,
            nominal=_pad_to_ndim(nominal, ndim),
            param_val=params,
        )

    @classmethod
    def batch_expressions(
        cls,
        members: Sequence[InterpolationFunction],
        context: Context,
        functions: Mapping[str, Function],
    ) -> dict[str, TensorVar]:
        """
        Evaluate many interpolation functions as one vectorised computation.

        Every entry of every member becomes one element of a flat entry axis
        of length ``E``, and the members one row of a ``(F, E_max)`` table:

        - Each entry's parameter is gathered from one stacked vector: a free
          parameter times a constant (a ``ProductFunction`` of one free
          parameter and constants, or a parameter itself), or else the entry's
          own expression with factor 1.
        - Entries are sorted by interpolation code, so each code's
          contributions are one contiguous slice, evaluated once with
          :meth:`_flexible_interp_single`, and the slices are concatenated.
        - Contributions are padded into ``(F, E_max)`` with a sentinel (factor
          1 for multiplicative, term 0 for additive members) and reduced over
          the entry axis.
        - The ``(F,)`` result is split into one scalar per member with a
          single ``Split``, so the gradient is one ``Join`` rather than a sum
          of ``F`` scattered ``IncSubtensor`` updates.

        Members whose inputs are not all scalars, or whose entries mix
        additive (0, 2, 3, 4) and multiplicative (1, 5, 6) codes, are built
        with their own :meth:`expression` and logged at debug level.

        Args:
            members: Interpolation functions to evaluate.
            context: Built tensors by name.
            functions: Every function of the model by name, to recognise
                ``ProductFunction`` entries.

        Returns:
            dict[str, TensorVar]: Scalar expression per member name.
        """
        outputs: dict[str, TensorVar] = {}
        rows: list[_InterpolationRow] = []
        for member in members:
            row = member._batch_row(context, functions)
            if isinstance(row, str):
                log.debug(
                    "Interpolation function '%s' is built on its own, not batched: %s",
                    member.name,
                    row,
                )
                outputs[member.name] = member.expression(context)
            else:
                rows.append(row)
        if rows:
            outputs.update(_batched_interpolation(rows))
        return outputs

    def _batch_row(
        self, context: Context, functions: Mapping[str, Function]
    ) -> _InterpolationRow | str:
        """This member's inputs for a batched evaluation, or why it cannot batch."""
        runs = self._entry_runs()
        if len(runs) > 1:
            return "its entries mix additive and multiplicative codes"
        nominal = context[self.nom]
        if nominal.ndim != 0:
            return f"its nominal '{self.nom}' is not a scalar"
        entries: list[_InterpolationEntry] = []
        for i in runs[0][1] if runs else []:
            low, high = context[self.low[i]], context[self.high[i]]
            if low.ndim != 0 or high.ndim != 0:
                return f"the variations of entry {i} are not scalars"
            source = _entry_source(self.vars[i], context, functions)
            if source is None:
                return f"entry '{self.vars[i]}' is not a scalar"
            entries.append(
                _InterpolationEntry(self.interpolationCodes[i], low, high, *source)
            )
        return _InterpolationRow(
            name=self.name,
            nominal=nominal,
            additive=bool(runs) and runs[0][0],
            positive_definite=self.positiveDefinite,
            entries=entries,
        )


@dataclass(frozen=True)
class _InterpolationEntry:
    """One entry: its code, variations, and parameter ``tensor * factor``."""

    code: int
    low: TensorVar
    high: TensorVar
    tensor: TensorVar
    factor: float


@dataclass(frozen=True)
class _InterpolationRow:
    """One member's inputs to :func:`_batched_interpolation`."""

    name: str
    nominal: TensorVar
    additive: bool
    positive_definite: bool
    entries: list[_InterpolationEntry]


def _entry_source(
    var: str, context: Context, functions: Mapping[str, Function]
) -> tuple[TensorVar, float] | None:
    """``(tensor, factor)`` with entry parameter ``tensor * factor``; None if not scalar.

    A ``ProductFunction`` of exactly one non-constant factor and constant
    factors gives that factor and the product of the constants, so that many
    entries share one gathered tensor.  Any other scalar entry is its own
    tensor with factor 1.
    """
    product = functions.get(var)
    if isinstance(product, ProductFunction):
        factors = product.get_parameter_list(context, "factors")
        if all(f.ndim == 0 for f in factors):
            free = [f for f in factors if not isinstance(f, TensorConstant)]
            if len(free) == 1:
                constants = [
                    float(f.data) for f in factors if isinstance(f, TensorConstant)
                ]
                return free[0], float(np.prod(constants))
    tensor = context[var]
    return (tensor, 1.0) if tensor.ndim == 0 else None


def _as_vector(tensors: Sequence[TensorVar]) -> npt.NDArray[np.float64] | TensorVar:
    """``(len(tensors),)`` values: a numpy table if every input is a constant, else stacked.

    Tables keep the per-entry bookkeeping (indexing, slicing) in numpy, out of
    the graph.
    """
    if all(isinstance(t, TensorConstant) for t in tensors):
        return np.array(
            [float(t.data) for t in tensors if isinstance(t, TensorConstant)],
            dtype=np.float64,
        )
    return cast(TensorVar, pt.stack(list(tensors)))


def _batched_interpolation(rows: Sequence[_InterpolationRow]) -> dict[str, TensorVar]:
    """The vectorised evaluation described in :meth:`InterpolationFunction.batch_expressions`."""
    n_rows = len(rows)
    nominal = _as_vector([row.nominal for row in rows])  # (F,)
    flat = sorted(
        ((r, entry) for r, row in enumerate(rows) for entry in row.entries),
        key=lambda item: item[1].code,
    )
    if flat:
        row_of = np.array([r for r, _ in flat])  # (E,) member of each entry
        codes = np.array([entry.code for _, entry in flat])
        # One gather: unique entry tensors stacked once, indexed per entry.
        slots: dict[int, int] = {}
        stacked: list[TensorVar] = []
        index = np.empty(len(flat), dtype=np.int64)
        for e, (_, entry) in enumerate(flat):
            key = id(entry.tensor)
            if key not in slots:
                slots[key] = len(stacked)
                stacked.append(entry.tensor)
            index[e] = slots[key]
        factor = np.array([entry.factor for _, entry in flat])
        params = cast(TensorVar, pt.stack(stacked)[index] * factor)  # (E,)
        lows = _as_vector([entry.low for _, entry in flat])  # (E,)
        highs = _as_vector([entry.high for _, entry in flat])  # (E,)
        entry_nominal = nominal[row_of]  # (E,)
        blocks = []
        for code in dict.fromkeys(codes.tolist()):
            sl = slice(*np.flatnonzero(codes == code)[[0, -1]] + [0, 1])
            blocks.append(
                InterpolationFunction._flexible_interp_single(
                    interp_code=code,
                    low_val=pt.as_tensor_variable(lows[sl]),
                    high_val=pt.as_tensor_variable(highs[sl]),
                    boundary=1.0,
                    nominal=pt.as_tensor_variable(entry_nominal[sl]),
                    param_val=params[sl],
                )
            )
        contributions = cast(TensorVar, pt.concatenate(blocks))  # type: ignore[no-untyped-call]  # (E,)
        # (F, E_max) positions into the entry axis; E (one past the end) is the
        # sentinel slot appended below.
        counts = np.bincount(row_of, minlength=n_rows)
        table = np.full((n_rows, counts.max()), len(flat), dtype=np.int64)
        position = np.zeros(n_rows, dtype=np.int64)
        for e, r in enumerate(row_of):
            table[r, position[r]] = e
            position[r] += 1
        additive = np.array([row.additive for row in rows])
        # Each reduction reads only its own kind's rows; rows of the other kind
        # read the sentinel alone, so e.g. an additive term of -1 never becomes
        # a zero factor in the product.
        multiplicative_table = np.where(additive[:, None], len(flat), table)
        additive_table = np.where(additive[:, None], table, len(flat))
        results = []
        if not additive.all():
            factors = pt.concatenate([1.0 + contributions, pt.ones(1)])[  # type: ignore[no-untyped-call]
                multiplicative_table
            ]
            results.append(
                nominal * pt.prod(factors, axis=1, no_zeros_in_input=True)  # type: ignore[no-untyped-call]
            )
        if additive.any():
            terms = pt.concatenate([contributions, pt.zeros(1)])[additive_table]  # type: ignore[no-untyped-call]
            results.append(nominal + pt.sum(terms, axis=1))  # type: ignore[no-untyped-call]
        result = (
            results[0]
            if len(results) == 1
            else pt.switch(additive, results[1], results[0])
        )
    else:
        result = pt.as_tensor_variable(nominal)
    positive = np.array([row.positive_definite for row in rows])
    if positive.any():
        result = pt.switch(positive, pt.maximum(result, 0.0), result)
    # One Split, not F Subtensors: the gradient of Split is one Join.
    pieces = (
        pt.split(result, [1] * n_rows, n_splits=n_rows)  # type: ignore[no-untyped-call]
        if n_rows > 1
        else [result]
    )
    return {
        row.name: cast(
            TensorVar,
            specify_shape(piece, (1,)).dimshuffle(()),
        )
        for row, piece in zip(rows, pieces, strict=True)
    }


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
    "generic_function": GenericFunction,
    "interpolation": InterpolationFunction,
    "CMS::process_normalization": ProcessNormalizationFunction,
    "CMS::asympow": CMSAsymPowFunction,
    "roorecursivefraction_dist": RooRecursiveFractionFunction,
}
