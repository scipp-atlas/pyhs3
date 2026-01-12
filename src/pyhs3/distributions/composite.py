"""
Composite distribution implementations.

Provides classes for handling composite probability distributions that combine
multiple other distributions, including mixtures and products of distributions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

import pytensor.tensor as pt
from pydantic import (
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_serializer,
)

from pyhs3.context import Context

# Import for Poisson extended likelihood calculation
from pyhs3.distributions.basic import PoissonDist
from pyhs3.distributions.core import Distribution
from pyhs3.typing.aliases import TensorVar


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

    Parameters:
        coefficients (list[str]): Names of coefficient parameters.
        summands (list[str]): Names of component distributions.
        extended (bool): Whether the mixture is extended (affects normalization).
            Must be True for N coefficients, False for N-1 coefficients.
        ref_coef_norm (list[str] | None): Optional list of coefficient names for custom normalization.
            Only valid when using N coefficients (extended=True).

    ROOT Reference:
        :rootref:`RooAddPdf <classRooAddPdf.html>`
    """

    type: Literal["mixture_dist"] = "mixture_dist"
    summands: list[str]
    coefficients: list[str]
    extended: bool = False
    ref_coef_norm: list[str] | None = Field(
        default=None, json_schema_extra={"preprocess": False}
    )

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
        elif n_coeffs == n_summands - 1:
            if extended:
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
            mixturesum = pt.constant(0.0)

            # Calculate the mixture sum
            for i, coeff in enumerate(self.coefficients):
                mixturesum += context[coeff] * context[self.summands[i]]

            # Handle normalization
            if self.ref_coef_norm is not None:
                # Custom normalization using specified coefficients
                norm_sum = pt.constant(0.0)
                for norm_coeff in self.ref_coef_norm:
                    norm_sum += context[norm_coeff]
                mixturesum = mixturesum / norm_sum
            else:
                # Standard normalization: divide by sum of all coefficients
                coeffsum = pt.constant(0.0)
                for coeff in self.coefficients:
                    coeffsum += context[coeff]
                mixturesum = mixturesum / coeffsum

        else:
            # N-1 coefficients case: traditional approach with automatic last term
            mixturesum = pt.constant(0.0)
            coeffsum = pt.constant(0.0)

            # Sum the first N-1 terms
            for i, coeff in enumerate(self.coefficients):
                coeffsum += context[coeff]
                mixturesum += context[coeff] * context[self.summands[i]]

            # Add the last term with remaining coefficient
            last_index = len(self.summands) - 1
            f_last = context[self.summands[last_index]]
            mixturesum += (1 - coeffsum) * f_last

        return cast(TensorVar, mixturesum)

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

        nu = pt.constant(0.0)

        if self.ref_coef_norm is not None:
            # Use only the coefficients specified in ref_coef_norm
            for coeff in self.ref_coef_norm:
                nu += context[coeff]
        else:
            # Use all coefficients
            for coeff in self.coefficients:
                nu += context[coeff]

        return cast(TensorVar, nu)

    def extended_likelihood(
        self, context: Context, data: TensorVar | None = None
    ) -> TensorVar:
        """
        Poisson term for the extended likelihood.

        Args:
            context: Mapping of names to pytensor variables
            data: Tensor containing observed event count (if None, returns 1.0)

        Returns:
            pytensor.tensor.variable.TensorVariable: :func:`PoissonDist.expression` object

        Raises:
            RuntimeError: If called on non-extended PDF
        """
        if not self.extended:
            return pt.constant(1.0)

        if data is None:
            # No data provided, return no contribution
            return pt.constant(1.0)

        nu = self.expected_yield(context)
        n_obs = data  # data should contain the observed count

        # Use the existing PoissonDist implementation for correctness
        poisson_dist = PoissonDist(name="temp_poisson", mean="nu", x="n_obs")
        poisson_context = Context(parameters={"nu": nu, "n_obs": n_obs})
        return poisson_dist.expression(poisson_context)


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

        pt_factors = pt.stack([context[factor] for factor in self.factors])
        return cast(TensorVar, pt.prod(pt_factors, axis=0))  # type: ignore[no-untyped-call]


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
