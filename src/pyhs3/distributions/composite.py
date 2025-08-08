"""
Composite distribution implementations.

Provides classes for handling composite probability distributions that combine
multiple other distributions, including mixtures and products of distributions.
"""

from __future__ import annotations

from typing import Literal, cast

import pytensor.tensor as pt

from pyhs3.context import Context
from pyhs3.distributions.core import Distribution
from pyhs3.typing.aliases import TensorVar


class MixtureDist(Distribution):
    r"""
    Mixture of probability distributions.

    Implements a weighted combination of multiple distributions:

    .. math::

        f(x) = \sum_{i=1}^{n-1} c_i \cdot f_i(x) + (1 - \sum_{i=1}^{n-1} c_i) \cdot f_n(x)

    The last component is automatically normalized to ensure the
    coefficients sum to 1.

    Parameters:
        coefficients (list[str]): Names of coefficient parameters.
        summands (list[str]): Names of component distributions.
        extended (bool): Whether the mixture is extended (affects normalization).
    """

    type: Literal["mixture_dist"] = "mixture_dist"
    summands: list[str]
    coefficients: list[str]
    extended: bool = False

    def expression(self, context: Context) -> TensorVar:
        """
        Builds a symbolic expression for the mixture distribution.

        Args:
            context (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of the mixture PDF.
        """

        mixturesum = pt.constant(0.0)
        coeffsum = pt.constant(0.0)

        for i, coeff in enumerate(self.coefficients):
            coeffsum += context[coeff]
            mixturesum += context[coeff] * context[self.summands[i]]

        last_index = len(self.summands) - 1
        f_last = context[self.summands[last_index]]
        mixturesum = mixturesum + (1 - coeffsum) * f_last
        return cast(TensorVar, mixturesum)


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

    def expression(self, context: Context) -> TensorVar:
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
