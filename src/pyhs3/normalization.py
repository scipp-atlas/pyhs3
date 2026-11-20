"""
Normalization utilities for HS3 distributions.

Provides Gauss-Legendre quadrature for computing normalization integrals symbolically.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytensor.tensor as pt
from pytensor.graph.basic import Variable
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import io_toposort

from pyhs3.typing.aliases import TensorVar

# Precompute 64-point Gauss-Legendre nodes and weights (fixed, deterministic)
_GL_ORDER = 64
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(_GL_ORDER)
_GL_NODES_T = pt.constant(_GL_NODES)
_GL_WEIGHTS_T = pt.constant(_GL_WEIGHTS)


def substitute_variable(
    output: TensorVar, variable: TensorVar, replacement: TensorVar
) -> TensorVar:
    """Return ``output`` with ``variable`` replaced by ``replacement``.

    Only the nodes that depend on ``variable`` are rebuilt, each through its
    ``Op``'s ``make_node`` so that its output type is inferred again from the
    new inputs; every other node is reused as it is.

    This differs from PyTensor's two built-in substitutions:

    - ``graph_replace`` rebuilds only the dependent nodes too, but keeps
      their original output types.  Replacing an observable of unknown
      length ``(?,)`` by a ``(1,)`` bound therefore leaves an axis typed as
      unknown that holds length 1, which PyTensor refuses to broadcast at
      runtime (https://github.com/scipp-atlas/pyhs3/issues/346).
    - ``clone_replace(..., rebuild_strict=False)`` infers the types again but
      copies the whole graph, including the nodes that never depended on
      ``variable``, so an antiderivative evaluated at two bounds carries
      two copies of its parameter-only parts.

    Args:
        output: Expression to substitute into.
        variable: Variable to replace (e.g. an observable's leaf vector).
        replacement: Variable to use instead; may have a different static
            shape, e.g. ``(1,)`` for ``(?,)``.

    Returns:
        The substituted expression, or ``output`` itself if it does not
        depend on ``variable``.
    """
    rebuilt: dict[Variable[Any, Any], Variable[Any, Any]] = {variable: replacement}
    for node in io_toposort([], [output]):
        if any(var in rebuilt for var in node.inputs):
            new_node = node.op.make_node(
                *(rebuilt.get(var, var) for var in node.inputs)
            )
            rebuilt.update(zip(node.outputs, new_node.outputs, strict=True))
    return cast(TensorVar, rebuilt.get(output, output))


def gauss_legendre_integral(
    expression: TensorVar,
    variable: TensorVar,  # note: variable must be the 1-D leaf tensor, not an ExpandDims view
    lower: TensorVar,
    upper: TensorVar,
) -> TensorVar:
    r"""
    Compute a definite integral symbolically via Gauss-Legendre quadrature.

    Gauss-Legendre quadrature approximates the integral over :math:`[a, b]` by
    evaluating the integrand at specific nodes and summing with weights. This
    implementation uses 64-point quadrature, which integrates polynomials up to
    degree 127 exactly.

    The standard Gauss-Legendre formula is defined on :math:`[-1, 1]`:

    .. math::

        \int_{-1}^{1} f(t)\,dt \approx \sum_{i=1}^{N} w_i\,f(t_i)

    To integrate over an arbitrary interval :math:`[a, b]`, we apply a linear
    change of variables:

    .. math::

        x = \frac{b-a}{2}t + \frac{a+b}{2}, \quad t \in [-1, 1], \quad x \in [a, b]

    The Jacobian :math:`dx = \frac{b-a}{2}dt` (the "half width") transforms the
    integral to:

    .. math::

        \int_a^b f(x)\,dx = \frac{b-a}{2} \int_{-1}^{1} f\left(\frac{b-a}{2}t + \frac{a+b}{2}\right) dt

    Applying Gauss-Legendre quadrature yields:

    .. math::

        \int_a^b f(x)\,dx \approx \frac{b-a}{2} \sum_{i=1}^{N} w_i\,f\left(\frac{b-a}{2}t_i + \frac{a+b}{2}\right)

    where :math:`t_i` are the standard Legendre nodes and :math:`w_i` are the
    standard weights on :math:`[-1, 1]`.

    See Also:
        https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature

    Args:
        expression: The PyTensor expression to integrate
        variable: The 1-D leaf tensor to integrate over (not an ExpandDims view).
            Passing the leaf means graph_replace propagates the substitution through
            any ExpandDims(variable) views present in the expression.
        lower: Lower bound :math:`a` (PyTensor expression)
        upper: Upper bound :math:`b` (PyTensor expression)

    Returns:
        Symbolic integral as a PyTensor expression
    """
    half_width = (upper - lower) / 2.0
    midpoint = (upper + lower) / 2.0

    # Quadrature nodes mapped to [lower, upper]: shape (64,)
    x_points = half_width * _GL_NODES_T + midpoint

    # Substitute the leaf with the (64,) quadrature points.  Any ExpandDims(variable)
    # views inside expression inherit the substitution and become (64, 1), so f_vals
    # may be (64,) or (64, ...).  Use strict=False so that constant integrands
    # (not depending on variable at all) are returned unchanged rather than raising.
    f_vals = cast(
        TensorVar,
        graph_replace(expression, replace=[(variable, x_points)], strict=False),
    )
    weights = _GL_WEIGHTS_T.reshape((-1,) + (1,) * (f_vals.ndim - 1))  # type: ignore[no-untyped-call]
    integral = half_width * pt.sum(weights * f_vals, axis=0)  # type: ignore[no-untyped-call]

    return cast(TensorVar, integral)
