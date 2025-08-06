"""
Core distribution classes and utilities.

Provides the base Distribution class and common utilities used by both
standard and CMS-specific distribution implementations.
"""

from __future__ import annotations

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

    def expression(self, _: Context) -> TensorVar:
        """
        Distribution-specific expression implementation.
        """
        msg = f"Distribution type={self.type} is not implemented."
        raise NotImplementedError(msg)
