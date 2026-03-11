"""
Normalization utilities for HS3 distributions.

Provides Gauss-Legendre quadrature for computing normalization integrals symbolically.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytensor.tensor as pt
from pytensor.graph.replace import clone_replace

from pyhs3.typing.aliases import TensorVar

# Precompute 64-point Gauss-Legendre nodes and weights (fixed, deterministic)
_GL_ORDER = 64
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(_GL_ORDER)
_GL_NODES_T = pt.constant(_GL_NODES)
_GL_WEIGHTS_T = pt.constant(_GL_WEIGHTS)


def gauss_legendre_integral(
    expression: TensorVar,
    variable: TensorVar,  # note: variable is expected to be a 1D vector (not a scalar)
    lower: TensorVar,
    upper: TensorVar,
) -> TensorVar:
    """
    Compute a definite integral symbolically via Gauss-Legendre quadrature.

    Builds a PyTensor expression: integral = (b-a)/2 * sum_i(w_i * f(x_i))
    using pytensor.scan.basic.scan for a compact symbolic loop.

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

    # evaluation points
    x_points = half_width * _GL_NODES_T + midpoint

    # directly replace the variable with the vector of points
    f_vals = clone_replace(expression, replace=[(variable, x_points)])

    # weighted sum
    integral = half_width * pt.sum(_GL_WEIGHTS_T * f_vals)  # type: ignore[no-untyped-call]

    return cast(TensorVar, integral)
