"""
Normalization utilities for HS3 distributions.

Provides Gauss-Legendre quadrature for computing normalization integrals symbolically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.graph.replace import clone_replace

if TYPE_CHECKING:
    from pyhs3.typing.aliases import TensorVar

# Precompute 64-point Gauss-Legendre nodes and weights (fixed, deterministic)
_GL_ORDER = 64
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(_GL_ORDER)


def gauss_legendre_integral(
    expression: TensorVar,
    variable: TensorVar,
    lower: TensorVar,
    upper: TensorVar,
) -> TensorVar:
    """
    Compute a definite integral symbolically via Gauss-Legendre quadrature.

    Builds a PyTensor expression: integral = (b-a)/2 * sum_i(w_i * f(x_i))
    using pytensor.scan for a compact symbolic loop.

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

    # Convert nodes and weights to tensors
    nodes = pt.as_tensor_variable(_GL_NODES)
    weights = pt.as_tensor_variable(_GL_WEIGHTS)
    x_points = half_width * nodes + midpoint

    def step(x_i: TensorVar, w_i: TensorVar) -> TensorVar:
        f_i = clone_replace(expression, replace={variable: x_i})  # type: ignore[arg-type]
        return cast("TensorVar", w_i * f_i)

    results = pytensor.scan(  # type: ignore[attr-defined]
        fn=step, sequences=[x_points, weights], return_updates=False
    )
    return cast("TensorVar", half_width * pt.sum(results))  # type: ignore[no-untyped-call]
