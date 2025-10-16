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
            NotImplementedError: Must be implemented by subclasses
        """
        msg = f"Distribution type={self.type} likelihood not implemented."
        raise NotImplementedError(msg)

    def expression(self, context: Context) -> TensorVar:
        """
        Complete probability combining main likelihood with extended terms.

        Returns the product of likelihood() and extended_likelihood().
        This provides the complete probability for the distribution.

        Subclasses typically do not need to override this method - just
        implement likelihood() and optionally extended_likelihood().

        Args:
            context: Mapping of names to pytensor variables

        Returns:
            TensorVar: Complete probability density
        """
        return cast(
            TensorVar, self.likelihood(context) * self.extended_likelihood(context)
        )

    def log_expression(self, context: Context) -> TensorVar:
        """
        Log-probability combining main likelihood with extended terms.

        Returns the sum of log(likelihood()) and log(extended_likelihood()).
        This is mathematically equivalent to log(likelihood * extended_likelihood)
        but can be more numerically stable.

        PyTensor handles optimization and simplification automatically.

        Args:
            context: Mapping of names to pytensor variables

        Returns:
            TensorVar: Log-probability density
        """
        return cast(
            TensorVar,
            pt.log(self.likelihood(context))
            + pt.log(self.extended_likelihood(context)),
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
