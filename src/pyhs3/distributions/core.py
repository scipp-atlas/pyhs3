"""
Core distribution classes and utilities.

Provides the base Distribution class and common utilities used by both
standard and CMS-specific distribution implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

import pytensor.tensor as pt
from pydantic import PrivateAttr
from pytensor.graph.replace import graph_replace

from pyhs3.base import Evaluable
from pyhs3.context import Context
from pyhs3.normalization import gauss_legendre_integral
from pyhs3.typing.aliases import TensorVar


class Distribution(Evaluable, ABC):
    """
    Base class for probability distributions in HS3.

    Provides the foundation for all distribution implementations,
    handling parameter management, constant generation, and symbolic
    expression evaluation using PyTensor.

    Distributions separate the main probability model (likelihood) from
    additional extended likelihood terms (e.g., constraints). The complete
    probability is the product of both terms.

    All distributions are automatically normalized over the domain of their
    observables unless explicitly opted out via _normalizable = False.

    Inherits parameter processing functionality from Evaluable.
    Subclasses must implement _expression() to define computation logic.
    """

    _normalizable: bool = PrivateAttr(default=True)

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

    def normalization_expression(
        self, _context: Context, _observable_name: str
    ) -> TensorVar | None:
        """
        Return the antiderivative expression, or None for numerical fallback.

        Override in subclasses to provide analytical normalization. The returned
        expression should be the antiderivative F(x) such that the integral
        ∫f(x)dx from a to b equals F(b) - F(a).

        Args:
            context: Mapping of names to pytensor variables
            observable_name: Name of the observable to integrate over

        Returns:
            Symbolic antiderivative expression, or None for numerical fallback.
        """
        return None

    def _normalization_integral(
        self, context: Context, obs_name: str, lower: TensorVar, upper: TensorVar
    ) -> TensorVar | None:
        """
        Evaluate normalization integral using the antiderivative expression.

        This is a private method that evaluates F(upper) - F(lower) where F is
        the antiderivative returned by normalization_expression().

        Args:
            context: Mapping of names to pytensor variables
            obs_name: Name of the observable to integrate over
            lower: Lower integration bound
            upper: Upper integration bound

        Returns:
            Symbolic integral expression, or None if normalization_expression() returns None.
        """
        expr = self.normalization_expression(context, obs_name)  # pylint: disable=assignment-from-none
        if expr is None:
            return None
        # Use the leaf (not the view) as the substitution target so graph_replace
        # propagates through every ExpandDims(leaf) view in the expression.
        leaf = context.parameters[obs_name]
        upper_t = pt.as_tensor_variable([upper], dtype=leaf.dtype)
        lower_t = pt.as_tensor_variable([lower], dtype=leaf.dtype)
        upper_val = cast(TensorVar, graph_replace(expr, [(leaf, upper_t)]))
        lower_val = cast(TensorVar, graph_replace(expr, [(leaf, lower_t)]))
        return cast(TensorVar, upper_val - lower_val)

    def _apply_normalization(
        self,
        raw: TensorVar,
        context: Context,
    ) -> TensorVar:
        """
        Apply normalization to a raw likelihood expression.

        Normalizes a likelihood over observables present in the context.
        Attempts analytical integration first via _normalization_integral(),
        then falls back to nested Gauss-Legendre quadrature for
        multi-dimensional integrals.

        Args:
            raw: Raw (unnormalized) likelihood expression
            context: Mapping of names to pytensor variables (includes observables)

        Returns:
            Normalized likelihood expression
        """
        # Explicit opt-out for distributions that should not be normalized
        if not self._normalizable:
            return raw

        matching = [
            (name, lower, upper)
            for name, (lower, upper) in context.observables.items()
            if name in self.parameters
        ]
        if not matching:
            return raw

        # Single observable: try analytical integral first
        if len(matching) == 1:
            obs_name, lower, upper = matching[0]
            integral = self._normalization_integral(context, obs_name, lower, upper)
            if integral is not None:
                return cast(TensorVar, raw / integral)

        if len(matching) > 1:
            obs_names = [name for name, _, _ in matching]
            msg = (
                f"Multi-dimensional normalization is not yet supported "
                f"(observables: {obs_names}). "
                f"See https://github.com/scipp-atlas/pyhs3/issues/214"
            )
            raise NotImplementedError(msg)

        # Single observable: fall back to Gauss-Legendre quadrature.
        # Pass the leaf (not the view) so graph_replace substitutes through
        # every ExpandDims(leaf) view inside the integrand.
        obs_name, lower, upper = matching[0]
        integral_expr = gauss_legendre_integral(
            raw, context.parameters[obs_name], lower, upper
        )
        return cast(TensorVar, raw / integral_expr)

    def _expression(self, context: Context) -> TensorVar:
        """
        Complete probability combining main likelihood with extended terms.

        Returns the product of likelihood() and extended_likelihood().
        This provides the complete probability for the distribution.

        All distributions are automatically normalized over observables present
        in the context, unless explicitly opted out via _normalizable = False.

        Subclasses typically do not need to override this method - just
        implement likelihood() and optionally extended_likelihood().

        Args:
            context: Mapping of names to pytensor variables

        Returns:
            TensorVar: Complete probability density
        """
        raw = self.likelihood(context)

        # Apply normalization (respects _normalizable flag internally)
        raw = self._apply_normalization(raw, context)

        return cast(TensorVar, raw * self.extended_likelihood(context))

    def log_expression(self, context: Context) -> TensorVar:
        """
        Log-probability combining main likelihood with extended terms.

        Returns the sum of log(likelihood()) and log(extended_likelihood()).
        This is mathematically equivalent to log(likelihood * extended_likelihood)
        but can be more numerically stable.

        All distributions are automatically normalized over observables present
        in the context, unless explicitly opted out via _normalizable = False.

        PyTensor handles optimization and simplification automatically.

        Args:
            context: Mapping of names to pytensor variables

        Returns:
            TensorVar: Log-probability density
        """
        raw = self.likelihood(context)

        # Apply normalization (respects _normalizable flag internally)
        raw = self._apply_normalization(raw, context)

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
