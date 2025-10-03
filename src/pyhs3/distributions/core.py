"""
Core distribution classes and utilities.

Provides the base Distribution class and common utilities used by both
standard and CMS-specific distribution implementations.
"""

from __future__ import annotations

from typing import cast

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

    def log_expression(self, context: Context) -> TensorVar:
        """
        Log-PDF expression in logarithmic space.

        Default implementation takes the logarithm of expression().
        PyTensor handles optimization and simplification automatically.
        Subclasses can override for custom log-space implementations.

        Args:
            context: Mapping of names to pytensor variables

        Returns:
            TensorVar: Log-probability density in log-space
        """
        return cast(TensorVar, pt.log(self.expression(context)))

    def extended_likelihood(
        self, _context: Context, _data: TensorVar | None = None
    ) -> TensorVar:
        """
        Extended likelihood contribution in normal space.

        Returns likelihood term for extended ML fitting.
        Override only when the distribution contributes extended terms.
        Default: no contribution (returns 1.0 in normal space).

        Args:
            context: Mapping of names to pytensor variables
            data: Optional data tensor for data-dependent terms

        Returns:
            TensorVar: Likelihood contribution (default: 1.0 = no contribution)
        """
        return pt.constant(1.0)
