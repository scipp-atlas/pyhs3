"""
Core distribution classes and utilities.

Provides the base Distribution class and common utilities used by both
standard and CMS-specific distribution implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

import pytensor.tensor as pt

from pyhs3.base import Evaluable
from pyhs3.context import Context
from pyhs3.normalization import Normalizable, gauss_legendre_integral
from pyhs3.typing.aliases import TensorVar


def _apply_normalization(
    raw: TensorVar,
    distribution: Distribution,
    context: Context,
) -> TensorVar:
    """
    Apply normalization to a raw likelihood expression.

    Helper function that normalizes a likelihood over observables present in
    the context. Attempts analytical integration first via normalization_integral(),
    then falls back to Gauss-Legendre quadrature.

    Args:
        raw: Raw (unnormalized) likelihood expression
        distribution: Distribution instance (for normalization_integral() method)
        context: Mapping of names to pytensor variables (includes observables)

    Returns:
        Normalized likelihood expression
    """
    normalized = raw
    for obs_name, (lower, upper) in context.observables.items():
        if obs_name not in distribution.parameters:
            continue
        # Try analytical integral from normalization_integral()
        integral = distribution.normalization_integral(context, obs_name, lower, upper)
        if integral is None:
            # Fall back to numerical quadrature
            obs_var = context[obs_name]
            integral = gauss_legendre_integral(raw, obs_var, lower, upper)
        normalized = normalized / integral
    return normalized


class Distribution(Evaluable, ABC):
    """
    Base class for probability distributions in HS3.

    Provides the foundation for all distribution implementations,
    handling parameter management, constant generation, and symbolic
    expression evaluation using PyTensor.

    Distributions separate the main probability model (likelihood) from
    additional extended likelihood terms (e.g., constraints). The complete
    probability is the product of both terms.

    Inherits parameter processing functionality from Evaluable.
    Subclasses must implement _expression() to define computation logic.
    """

    @abstractmethod
    def likelihood(self, context: Context) -> TensorVar:
        """
        Main probability model for the distribution.

        This is the core probability density function (PDF) for the distribution.
        For example, the Poisson probability for observed data, Gaussian PDF, etc.
        Must be implemented by all subclasses.

        Args:
            context: Mapping of names to pytensor variables

        Returns:
            TensorVar: Main probability density

        Raises:
            TypeError: Must be implemented by subclasses
        """

    def normalization_integral(
        self,
        context: Context,  # noqa: ARG002
        observable_name: str,  # noqa: ARG002
        lower: TensorVar,  # noqa: ARG002
        upper: TensorVar,  # noqa: ARG002
    ) -> TensorVar | None:
        """
        Analytical normalization integral over the observable domain.

        Override in subclasses to provide known analytical integrals.
        Return None (default) to use Gauss-Legendre quadrature fallback.

        This method is only called when the distribution inherits from Normalizable.

        Args:
            context: Mapping of names to pytensor variables
            observable_name: Name of the observable to integrate over
            lower: Lower integration bound
            upper: Upper integration bound

        Returns:
            Symbolic integral expression, or None for numerical fallback.
        """
        return None

    def _expression(self, context: Context) -> TensorVar:
        """
        Complete probability combining main likelihood with extended terms.

        Returns the product of likelihood() and extended_likelihood().
        This provides the complete probability for the distribution.

        For distributions inheriting from Normalizable, applies normalization
        over observables present in the context.

        Subclasses typically do not need to override this method - just
        implement likelihood() and optionally extended_likelihood().

        Args:
            context: Mapping of names to pytensor variables

        Returns:
            TensorVar: Complete probability density
        """
        raw = self.likelihood(context)

        # Apply normalization if this distribution is Normalizable
        if isinstance(self, Normalizable):
            raw = _apply_normalization(raw, self, context)

        return cast(TensorVar, raw * self.extended_likelihood(context))

    def log_expression(self, context: Context) -> TensorVar:
        """
        Log-probability combining main likelihood with extended terms.

        Returns the sum of log(likelihood()) and log(extended_likelihood()).
        This is mathematically equivalent to log(likelihood * extended_likelihood)
        but can be more numerically stable.

        For distributions inheriting from Normalizable, applies normalization
        over observables present in the context.

        PyTensor handles optimization and simplification automatically.

        Args:
            context: Mapping of names to pytensor variables

        Returns:
            TensorVar: Log-probability density
        """
        raw = self.likelihood(context)

        # Apply normalization if this distribution is Normalizable
        if isinstance(self, Normalizable):
            raw = _apply_normalization(raw, self, context)

        return cast(
            TensorVar,
            pt.log(raw) + pt.log(self.extended_likelihood(context)),
        )

    def extended_likelihood(
        self, _context: Context, _data: TensorVar | None = None
    ) -> TensorVar:
        """
        Extended likelihood contribution in normal space.

        Returns additional likelihood terms for extended ML fitting.
        Override only when the distribution contributes extended terms like
        constraint terms (HistFactory) or Poisson yield terms (MixtureDist).

        Default: no contribution (returns 1.0 in normal space).

        Args:
            context: Mapping of names to pytensor variables
            data: Optional data tensor for data-dependent terms

        Returns:
            TensorVar: Likelihood contribution (default: 1.0 = no contribution)
        """
        return pt.constant(1.0)
