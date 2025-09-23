"""
Core distribution classes and utilities.

Provides the base Distribution class and common utilities used by both
standard and CMS-specific distribution implementations.
"""

from __future__ import annotations

import pytensor.tensor as pt

from pyhs3.base import Evaluable
from pyhs3.context import Context
from pyhs3.typing.aliases import TensorVar


class Distribution(Evaluable):
    """
    Base class for probability distributions in HS3.

    Provides the foundation for all distribution implementations,
    handling parameter management, constant generation, and symbolic
    expression evaluation using PyTensor.

    Inherits parameter processing functionality from Evaluable.
    """

    def log_expression(self, _context: Context) -> TensorVar:
        """
        Log-PDF expression in logarithmic space (PRIMARY METHOD).

        All distributions should implement their core logic here for numerical stability.
        This is the primary method for probability distributions.

        Args:
            _context: Mapping of names to pytensor variables

        Returns:
            TensorVar: Log-probability density in log-space

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        msg = f"log_expression not implemented for {self.type}"
        raise NotImplementedError(msg)

    def extended_likelihood(
        self, _context: Context, _data: TensorVar | None = None
    ) -> TensorVar:
        """
        Extended likelihood contribution in log-space.

        Returns log-likelihood term for extended ML fitting.
        Override only when the distribution contributes extended terms.
        Default: no contribution (returns 0.0 in log-space).

        Args:
            context: Mapping of names to pytensor variables
            data: Optional data tensor for data-dependent terms

        Returns:
            TensorVar: Log-likelihood contribution (default: 0.0 = no contribution)
        """
        return pt.constant(0.0)

    def expression(self, _context: Context) -> TensorVar:
        """
        Distribution-specific expression implementation.

        Note: This method will eventually be derived from log_expression().
        For now, subclasses should implement this method directly.

        Args:
            _context: Mapping of names to pytensor variables

        Returns:
            TensorVar: Distribution expression

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        msg = f"Distribution type={self.type} is not implemented."
        raise NotImplementedError(msg)
