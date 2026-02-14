"""
Normalization utilities for HS3 distributions.

Provides the Normalizable mixin for distributions that need numerical normalization,
and Gauss-Legendre quadrature for computing normalization integrals symbolically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytensor.tensor as pt
from pytensor.graph.basic import clone_replace

if TYPE_CHECKING:
    from pyhs3.context import Context
    from pyhs3.typing.aliases import TensorVar

# Precompute 64-point Gauss-Legendre nodes and weights (fixed, deterministic)
_GL_ORDER = 64
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(_GL_ORDER)


class Normalizable:
    """
    Mixin for distributions that need numerical normalization over observables.

    Distributions that inherit from this mixin will have their likelihood
    automatically divided by the integral over observable domains when
    observables are present in the Context.

    Subclasses can override normalization_integral() to provide analytical
    normalization formulas instead of the numerical fallback.
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

        Args:
            context: Mapping of names to pytensor variables
            observable_name: Name of the observable to integrate over
            lower: Lower integration bound
            upper: Upper integration bound

        Returns:
            Symbolic integral expression, or None for numerical fallback.
        """
        return None


def gauss_legendre_integral(
    expression: TensorVar,
    variable: TensorVar,
    lower: TensorVar,
    upper: TensorVar,
) -> TensorVar:
    """
    Compute a definite integral symbolically via Gauss-Legendre quadrature.

    Builds a PyTensor expression: integral = (b-a)/2 * sum_i(w_i * f(x_i))
    using clone_replace to substitute the variable at each quadrature node.

    Args:
        expression: The PyTensor expression to integrate
        variable: The PyTensor variable to integrate over
        lower: Lower bound (PyTensor expression)
        upper: Upper bound (PyTensor expression)

    Returns:
        Symbolic integral as a PyTensor expression
    """
    half_width = (upper - lower) / 2.0
    midpoint = (upper + lower) / 2.0

    integral = pt.constant(0.0)
    for node, weight in zip(_GL_NODES, _GL_WEIGHTS, strict=False):
        x_i = half_width * pt.constant(float(node)) + midpoint
        f_i = clone_replace(expression, replace={variable: x_i})
        integral = integral + pt.constant(float(weight)) * f_i

    return cast("TensorVar", half_width * integral)
