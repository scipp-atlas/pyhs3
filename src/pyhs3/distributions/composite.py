"""
Composite distribution implementations.

Provides classes for handling composite probability distributions that combine
multiple other distributions, including mixtures and products of distributions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

import pytensor.tensor as pt
from pydantic import (
    Field,
    PrivateAttr,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_serializer,
)
from pytensor.graph.traversal import explicit_graph_inputs

from pyhs3.base import balanced_sum
from pyhs3.context import Context
from pyhs3.distributions.core import Distribution, LogProbTerms
from pyhs3.typing.aliases import TensorVar

if TYPE_CHECKING:
    from pyhs3.distributions import Distributions


def _stable_logsumexp(x: TensorVar, axis: int) -> TensorVar:
    """Numerically stable log-sum-exp, built explicitly instead of via ``pt.logsumexp``.

    ``pt.logsumexp`` lowers to the literal ``log(sum(exp(x)))`` graph; the
    protection against overflow/underflow is not part of that graph but is
    instead an emergent property of the ``local_log_sum_exp`` graph rewrite,
    which only runs in the stabilize/specialize passes under ``FAST_RUN``.
    Any mode that skips those passes (e.g. ``FAST_COMPILE``, which
    ``Model``/``Workspace.model()`` expose to callers via ``mode=``) silently
    loses the protection. Building the max-shifted form here makes the
    stability structural rather than rewrite-dependent. See
    https://github.com/scipp-atlas/pyhs3/issues/277 for the failure mode and
    https://github.com/pymc-devs/pytensor/issues/2288 for the tracked
    upstream request for a stable-by-construction ``logsumexp``.

    Args:
        x: Tensor to reduce.
        axis: Axis to reduce over.

    Returns:
        log(sum(exp(x), axis=axis)), stable against underflow (large
        negative x, the case relevant to :class:`MixtureDist`) and overflow
        (large positive x). Reduces to -inf (not NaN) when every entry along
        axis is -inf: the ``pt.isinf`` guard below substitutes 0.0 for the
        max in that slice so the subsequent subtraction is ``-inf - 0.0``
        rather than ``-inf - -inf`` (NaN). The *value* is finite there, but
        its gradient is not: d/ds log(s) diverges as s = sum(exp(x - safe_m))
        approaches 0, so the gradient at an all-(-inf) point is NaN. That is
        a genuine singularity of log at its domain boundary -- present in
        any logsumexp implementation, not an artifact of the guard above
        (whose own gradient contribution stays finite; see the
        ``_stable_logsumexp`` tests in ``tests/test_distributions.py``).
    """
    m = pt.max(x, axis=axis, keepdims=True)  # type: ignore[no-untyped-call]
    safe_m = pt.switch(pt.isinf(m), 0.0, m)
    result = safe_m + pt.log(
        pt.sum(pt.exp(x - safe_m), axis=axis, keepdims=True)  # type: ignore[no-untyped-call]
    )
    return cast(TensorVar, pt.squeeze(result, axis=axis))  # type: ignore[no-untyped-call]


class MixtureDist(Distribution):
    r"""
    Mixture of probability distributions.

    Implements a weighted combination of multiple distributions following ROOT's RooAddPdf.
    Supports both N and N-1 coefficient configurations where :math:`N` represents number of distributions (`summands`):

    **N-1 coefficients:**

    .. math::

        f(x) = \sum_{i=1}^{n-1} c_i \cdot f_i(x) + (1 - \sum_{i=1}^{n-1} c_i) \cdot f_n(x)

    **N coefficients:**

    .. math::

        f(x) = \frac{\sum_{i=1}^{n} c_i \cdot f_i(x)}{\sum_{i=1}^{n} c_i}

    **N coefficients with `ref_coef_norm`:**

    .. math::

        f(x) = \frac{\sum_{i=1}^{n} c_i \cdot f_i(x)}{\sum_{j \in \text{ref\_coef\_norm}} c_j}

    :meth:`log_prob_terms` is the log-space-safe path for an extended mixture:
    it builds log(Σcᵢfᵢ) via logsumexp over each summand's own log-space
    expression rather than pt.log of the probability-space sum, so it stays
    finite even when every component underflows to 0.0.  The base-class
    :meth:`~pyhs3.distributions.core.Distribution.log_expression` (used for
    non-extended mixtures and standalone evaluation) still round-trips
    through probability-space component tensors and cannot be converted
    until composite distributions receive log-space component context (see
    https://github.com/scipp-atlas/pyhs3/issues/254, phase 4).

    Parameters:
        coefficients (list[str]): Names of coefficient parameters.
        summands (list[str]): Names of component distributions.
        extended (bool): Whether the mixture is extended (affects normalization).
            Must be True for N coefficients, False for N-1 coefficients.
        ref_coef_norm (list[str] | None): Optional list of coefficient names for custom normalization.
            Only valid when using N coefficients (extended=True).

    ROOT Reference:
        :root:`RooAddPdf`
    """

    type: Literal["mixture_dist"] = "mixture_dist"
    summands: list[str]
    coefficients: list[str]
    extended: bool = False
    ref_coef_norm: list[str] | None = Field(
        default=None, json_schema_extra={"preprocess": False}
    )

    # PyTensor nodes built by the most recent likelihood() call, reused by
    # unnormalized_expression() and expected_yield() so that model.log_prob
    # shares the same subgraphs instead of building duplicates the compiler
    # then has to merge (a measurable compile-time cost on large workspaces).
    _cached_unnorm_expr: TensorVar | None = PrivateAttr(default=None)
    _cached_nu_expr: TensorVar | None = PrivateAttr(default=None)

    # Per-summand coefficient nodes (context[coeff] for each entry in
    # self.coefficients, N==N case only), cached alongside _cached_unnorm_expr
    # so that log_prob_terms can build the log-space logsumexp directly from
    # the same coefficient tensors used in the probability-space sum, rather
    # than re-resolving them from a context it is not given.
    _cached_coeffs: list[TensorVar] | None = PrivateAttr(default=None)

    @model_serializer(mode="wrap")
    def serialize_model(self, handler: Callable[[Any], Any]) -> Any:
        """Do not serialize ref_coef_norm if it is unspecified (None)."""
        data = handler(self)
        if self.ref_coef_norm is None:
            del data["ref_coef_norm"]
        return data

    @field_validator("ref_coef_norm", mode="before")
    @classmethod
    def split_comma_separated_ref_coef_norm(cls, v: object) -> object:
        """Convert comma-separated string to list for ref_coef_norm."""
        if isinstance(v, str):
            v = v.strip()
            return None if v == "" else v.split(",")
        return v

    @field_validator("ref_coef_norm", mode="after")
    @classmethod
    def validate_ref_coef_norm_usage(
        cls, ref_coef_norm: list[str] | None, info: ValidationInfo
    ) -> list[str] | None:
        """Validate that ref_coef_norm is only used with N=N coefficient case."""
        if ref_coef_norm is not None:
            # Get summands and coefficients from the values being validated
            summands = info.data.get("summands", [])
            coefficients = info.data.get("coefficients", [])
            n_coeffs = len(coefficients)
            n_summands = len(summands)

            if n_coeffs != n_summands:
                msg = (
                    f"ref_coef_norm can only be used with N coefficients and N summands "
                    f"(N={n_summands}), but got {n_coeffs} coefficients."
                )
                raise ValueError(msg)

        return ref_coef_norm

    @field_serializer("ref_coef_norm")
    def serialize_ref_coef_norm(self, ref_coef_norm: list[str] | None) -> str | None:
        """Convert list back to comma-separated string for serialization."""
        if ref_coef_norm is None:
            return None
        return ",".join(ref_coef_norm)

    @field_validator("coefficients", mode="after")
    @classmethod
    def validate_coefficient_count(
        cls, coefficients: list[str], info: ValidationInfo
    ) -> list[str]:
        """Validate that coefficient count matches summand count appropriately."""
        # Get summands from the values being validated
        summands = info.data.get("summands", [])
        n_coeffs = len(coefficients)
        n_summands = len(summands)

        if n_coeffs not in (n_summands, n_summands - 1):
            msg = (
                f"Invalid coefficient configuration: {n_coeffs} coefficients "
                f"for {n_summands} summands. Must have N ({n_summands}) or "
                f"N-1 ({n_summands - 1}) coefficients."
            )
            raise ValueError(msg)

        return coefficients

    @field_validator("extended", mode="after")
    @classmethod
    def validate_extended_matches_coefficients(
        cls, extended: bool, info: ValidationInfo
    ) -> bool:
        """Validate that extended matches coefficient configuration."""
        # Get summands and coefficients from the values being validated
        summands = info.data.get("summands", [])
        coefficients = info.data.get("coefficients", [])
        n_coeffs = len(coefficients)
        n_summands = len(summands)

        # Validate extended matches coefficient configuration
        if n_coeffs == n_summands:
            if not extended:
                msg = (
                    f"extended must be True when N coefficients = N summands "
                    f"({n_coeffs} coefficients, {n_summands} summands)."
                )
                raise ValueError(msg)
        elif extended:
            msg = (
                f"extended must be False when N-1 coefficients with N summands "
                f"({n_coeffs} coefficients, {n_summands} summands)."
            )
            raise ValueError(msg)

        return extended

    def likelihood(self, context: Context) -> TensorVar:
        """
        Builds a symbolic expression for the mixture distribution.

        Handles both N and N-1 coefficient cases:
        - N-1 coefficients: Traditional approach with automatic normalization
        - N coefficients: Direct summation with optional custom normalization

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of the mixture PDF.
        """
        n_coeffs = len(self.coefficients)
        n_summands = len(self.summands)

        if n_coeffs == n_summands:
            # N coefficients case: direct summation with normalization

            # Calculate the mixture sum (unnormalized: Σ cᵢ fᵢ)
            coeff_tensors = [context[coeff] for coeff in self.coefficients]
            mixture_terms = [
                coeff_tensor * context[summand]
                for coeff_tensor, summand in zip(
                    coeff_tensors, self.summands, strict=True
                )
            ]
            mixturesum = balanced_sum(mixture_terms, pt.constant(0.0))

            # Cache the unnormalized Σcᵢfᵢ so that model.log_prob can build
            # the extended log-likelihood as Σⱼ log(Σcᵢfᵢ(xⱼ)) - nu without
            # introducing a separate pt.log(nu) node that triggers costly
            # optimizer rewrites when combined with log(Σcᵢfᵢ/nu).
            self._cached_unnorm_expr = mixturesum
            # Cache the per-summand coefficient nodes (same order as
            # self.summands) so log_prob_terms can build the log-space
            # equivalent Σᵢ cᵢfᵢ = logsumexp(log cᵢ + log fᵢ) directly, which
            # stays finite where the probability-space fᵢ underflow to 0.0.
            self._cached_coeffs = coeff_tensors

            # Handle normalization
            if self.ref_coef_norm is not None:
                # Custom normalization using specified coefficients
                norm_sum = balanced_sum(
                    [context[norm_coeff] for norm_coeff in self.ref_coef_norm],
                    pt.constant(0.0),
                )
                # Cache the normalisation sum so expected_yield() can reuse
                # the same PyTensor node rather than building a duplicate
                # summation subgraph that the compiler then has to merge.
                self._cached_nu_expr = norm_sum
                mixturesum = mixturesum / norm_sum
            else:
                # Standard normalization: divide by sum of all coefficients
                coeffsum = balanced_sum(
                    [context[coeff] for coeff in self.coefficients],
                    pt.constant(0.0),
                )
                # Cache so expected_yield() reuses this node.
                self._cached_nu_expr = coeffsum
                mixturesum = mixturesum / coeffsum

        else:
            # N-1 coefficients case: traditional approach with automatic last term
            coeffsum = balanced_sum(
                [context[coeff] for coeff in self.coefficients], pt.constant(0.0)
            )

            # First N-1 terms
            mixture_terms = [
                context[coeff] * context[self.summands[i]]
                for i, coeff in enumerate(self.coefficients)
            ]

            # Add the last term with remaining coefficient
            last_index = len(self.summands) - 1
            f_last = context[self.summands[last_index]]
            mixture_terms.append((1 - coeffsum) * f_last)

            mixturesum = balanced_sum(mixture_terms, pt.constant(0.0))

        return mixturesum

    def expected_yield(self, context: Context) -> TensorVar:
        """
        Compute the total expected yield nu in the extended case.

        - N coefficients case: nu = sum(coefficients) or sum(ref_coef_norm) if specified
        - N-1 coefficients case: not defined (extended=False always)

        Args:
            context: Mapping of names to pytensor variables

        Returns:
            Expected yield (nu) for extended likelihood

        Raises:
            RuntimeError: If called on non-extended PDF
        """
        if not self.extended:
            msg = "expected_yield only valid for extended PDFs"
            raise RuntimeError(msg)

        # Reuse the normalisation sum cached by likelihood() to avoid building
        # a duplicate PyTensor summation subgraph (which would slow compilation).
        if self._cached_nu_expr is not None:
            return self._cached_nu_expr

        # Fallback: likelihood() hasn't been called yet, compute fresh.
        coeff_names = (
            self.ref_coef_norm if self.ref_coef_norm is not None else self.coefficients
        )
        return balanced_sum([context[coeff] for coeff in coeff_names], pt.constant(0.0))

    def unnormalized_expression(self, context: Context) -> TensorVar:
        """Return the unnormalized mixture Σcᵢfᵢ (before dividing by Σcᵢ).

        After :meth:`likelihood` has been called this just returns the cached
        intermediate result.  This is a probability-space quantity: for the
        per-event log-likelihood, :meth:`log_prob_terms` builds
        log(Σcᵢfᵢ(xⱼ)) directly from each summand's own log-space expression
        via logsumexp instead of taking ``pt.log`` of this sum, so that the
        result stays finite where the probability-space fᵢ(xⱼ) underflow to
        0.0.  This method remains available for callers wanting the raw
        probability-space value itself.

        The Poisson yield term enters the likelihood once per channel and
        involves the observed event count, so it is assembled by
        :meth:`pyhs3.model.Model.log_prob` (which owns the channel-dataset
        pairing) rather than via :meth:`extended_likelihood` (whose result is
        multiplied into the per-event density and would be overcounted when
        summing over events).
        """
        if self._cached_unnorm_expr is not None:
            return self._cached_unnorm_expr

        # Fallback: recompute (likelihood() hasn't been called yet).
        unnorm_terms = [
            context[coeff] * context[self.summands[i]]
            for i, coeff in enumerate(self.coefficients)
        ]
        return balanced_sum(unnorm_terms, pt.constant(0.0))

    def log_prob_terms(  # pylint: disable=redefined-outer-name
        self,
        expressions: Mapping[str, TensorVar],
        log_expressions: Mapping[str, TensorVar],
        distributions: Distributions,
    ) -> LogProbTerms:
        """
        Extended-likelihood contributions for the mixture.

        For an extended mixture the per-channel log-likelihood is built from
        the unnormalized mixture and the expected yield:

            Σⱼ log(Σᵢcᵢfᵢ(xⱼ))  -  nu

        which simplifies from Σⱼ log(Σᵢcᵢfᵢ(xⱼ)/nu) + N·log(nu) - nu (the
        log(nu) terms cancel).  This is algebraically identical to the full
        extended likelihood but avoids introducing pt.log(nu) into the graph,
        preventing costly optimizer rewrite cascades triggered by
        log(a/b) + N·log(b).  For weighted data this form also reproduces
        RooFit's sum-of-weights convention: weighting the log(Σᵢcᵢfᵢ) term
        gives Σⱼwⱼ·log(PDF) + (Σwⱼ)·log(nu) - nu.

        log(Σᵢcᵢfᵢ(xⱼ)) is evaluated as logsumexp(log cᵢ + log fᵢ(xⱼ)) using
        each summand's own log-space expression (``log_expressions``) rather
        than ``pt.log`` of the probability-space sum
        (:attr:`_cached_unnorm_expr`).  fᵢ(xⱼ) can underflow to 0.0 in
        float64 for xⱼ far in a component's tail, which would make every
        cᵢfᵢ(xⱼ) — and hence their sum — exactly 0.0 and ``pt.log`` of it
        ``-inf``, even though log fᵢ(xⱼ) itself is finite.  Building the sum
        in log space keeps it finite in that regime.  The log cᵢ + log fᵢ(xⱼ)
        terms are stacked on a new leading axis (shape ``(n_summands, N, M)``
        for ``N`` events and parameter-batch size ``M``, matching the
        per-event shape documented on :class:`LogProbTerms`) and reduced with
        :func:`_stable_logsumexp` over that axis, giving the ``(N, M)``
        per-event term.  :func:`_stable_logsumexp` is used instead of
        ``pt.logsumexp`` because the latter's stability is an emergent
        property of a graph rewrite that only runs under ``FAST_RUN`` (see
        :func:`_stable_logsumexp`'s docstring and
        https://github.com/scipp-atlas/pyhs3/issues/277).  This assumes cᵢ ≥ 0, matching the probabilistic meaning of a mixture
        weight; :class:`MixtureDist` does not validate coefficient sign, so a
        negative cᵢ produces NaN here (``pt.log`` of a negative number) just
        as ``pt.log(Σᵢcᵢfᵢ)`` would if the probability-space sum itself were
        negative.

        The data-only -log(N!) constant is omitted and N_eff = Σwⱼ is used
        for weighted data; both are RooFit conventions adopted because HS3
        does not specify the extended term exactly.  See
        https://github.com/hep-statistics-serialization-standard/hep-statistics-serialization-standard/issues/91
        and https://github.com/scipp-atlas/pyhs3/issues/241.

        Non-extended mixtures contribute the default per-event log(PDF).
        """
        if not self.extended:
            return super().log_prob_terms(expressions, log_expressions, distributions)

        if (
            self._cached_unnorm_expr is None
            or self._cached_nu_expr is None
            or self._cached_coeffs is None
        ):
            msg = (
                f"log_prob_terms for extended mixture '{self.name}' requires "
                f"likelihood() to have been called (the model graph build does "
                f"this) so the unnormalized mixture and yield nodes exist."
            )
            raise RuntimeError(msg)

        log_terms = [
            pt.log(coeff) + log_expressions[summand]
            for coeff, summand in zip(self._cached_coeffs, self.summands, strict=True)
        ]
        per_event_term = _stable_logsumexp(pt.stack(log_terms, axis=0), axis=0)

        return LogProbTerms(
            per_event=[per_event_term],
            channel=[-self._cached_nu_expr],
        )


class ProductDist(Distribution):
    r"""
    Product distribution implementation.

    Implements a product of PDFs as defined in ROOT's RooProdPdf.

    The probability density function is defined as:

    .. math::

        f(x, \ldots) = \prod_{i=1}^{N} \text{PDF}_i(x, \ldots)

    where each PDF_i is a component distribution that may share observables.

    Parameters:
        factors: List of component distribution names to multiply together

    Note:
        In the context of pytensor variables/tensors, this is implemented as
        an elementwise product of all factor distributions.
    """

    type: Literal["product_dist"] = "product_dist"
    factors: list[str]

    def likelihood(self, context: Context) -> TensorVar:
        """
        Evaluate the product distribution.

        Args:
            context: Mapping of names to pytensor variables

        Returns:
            Symbolic representation of the product PDF
        """
        if not self.factors:
            return cast(TensorVar, pt.constant(1.0))

        vals = [context[factor] for factor in self.factors]
        return cast(TensorVar, pt.mul(*vals))

    def log_prob_terms(  # pylint: disable=redefined-outer-name
        self,
        expressions: Mapping[str, TensorVar],
        log_expressions: Mapping[str, TensorVar],
        distributions: Distributions,
    ) -> LogProbTerms:
        """
        Split factors into per-event shape terms and per-channel constraints.

        Factors that depend on an observable (detected by a pt.vector free
        input — observable arrays are built as pt.vector, NP scalars as
        pt.scalar) delegate to their own log_prob_terms, so e.g. an extended
        mixture inside a product contributes its yield term correctly.

        Factors that depend only on scalar nuisance parameters are constraint
        PDFs and are returned as named constraints for the model to add once
        globally.  This prevents the N-fold overcounting that occurs when
        naively summing log(shape x Π_j constr_j) over N events, which
        multiplies each constr_j term by N rather than counting it once.

        The shape-vs-constraint classification is structural inference
        (matching RooProdPdf), not HS3 spec semantics — nothing in an HS3
        file marks a factor as a constraint.  If the spec adopts explicit
        constraint encoding, this inference can be removed; see
        https://github.com/hep-statistics-serialization-standard/hep-statistics-serialization-standard/issues/90
        and https://github.com/scipp-atlas/pyhs3/issues/240.
        """
        terms = LogProbTerms()
        for factor_name in self.factors:
            factor_expr = expressions[factor_name]
            free_inputs = [
                v for v in explicit_graph_inputs([factor_expr]) if v.name is not None
            ]
            # Observable-dependent factors have at least one vector (rank-1) input.
            is_shape = any(v.type.ndim >= 1 for v in free_inputs)

            if is_shape:
                factor_terms = distributions[factor_name].log_prob_terms(
                    expressions, log_expressions, distributions
                )
                terms.per_event.extend(factor_terms.per_event)
                terms.channel.extend(factor_terms.channel)
                terms.constraints.update(factor_terms.constraints)
            else:
                terms.constraints[factor_name] = log_expressions[factor_name]
        return terms


# Registry of composite distributions
distributions: dict[str, type[Distribution]] = {
    "mixture_dist": MixtureDist,
    "product_dist": ProductDist,
}

# Define what should be exported from this module
__all__ = [
    "MixtureDist",
    "ProductDist",
    "distributions",
]
