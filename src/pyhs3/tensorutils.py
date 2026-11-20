from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytensor.tensor as pt
from pytensor.graph.basic import Apply, Constant, Variable
from pytensor.graph.traversal import io_toposort
from pytensor.tensor.variable import TensorConstant

from pyhs3.typing.aliases import DomainBounds, TensorVar


def is_scalar_multiplicative_identity(value: TensorVar) -> bool:
    """Return whether a tensor is a scalar constant equal to one."""
    return (
        isinstance(value, TensorConstant) and value.ndim == 0 and value.data.item() == 1
    )


def merge_duplicate_nodes(output: TensorVar) -> TensorVar:
    """Return ``output`` with every structurally identical subgraph built once.

    Separately built HS3 objects often produce the same subexpression: for
    example, several summands of a mixture that share their shape parameters
    each build the same normalised density.  PyTensor's merge rewrite removes
    such duplicates during compilation, but it does so one node at a time
    inside the rewriter.  This pass does the same merge in a single
    bottom-up sweep before compilation: an apply node is kept only if no
    earlier node has the same ``Op`` applied to the same (already merged)
    inputs, and constants are identified by type and value.

    Graph inputs (parameters and observables) are kept as the same variable
    objects, so names and :func:`~pytensor.graph.traversal.explicit_graph_inputs`
    are unchanged.

    Args:
        output: Output variable of the graph to merge.

    Returns:
        A variable computing the same value from the same inputs.  It is
        ``output`` itself when the graph contains nothing to merge.
    """
    replacement: dict[Variable[Any, Any], Variable[Any, Any]] = {}
    constants: dict[tuple[object, ...], Variable[Any, Any]] = {}
    kept: dict[tuple[object, ...], Apply[Any]] = {}

    def resolve(var: Variable[Any, Any]) -> Variable[Any, Any]:
        if var in replacement:
            return replacement[var]
        if isinstance(var, Constant):
            data = np.asarray(var.data)
            key = (var.type, data.shape, data.tobytes())
            replacement[var] = constants.setdefault(key, var)
            return replacement[var]
        return var

    for node in io_toposort([], [output]):
        inputs = [resolve(var) for var in node.inputs]
        key = (node.op, tuple(inputs))
        match = kept.get(key)
        if match is None:
            changed = any(
                new is not old for new, old in zip(inputs, node.inputs, strict=True)
            )
            match = node.clone_with_new_inputs(inputs) if changed else node
            kept[key] = match
        for old, new in zip(node.outputs, match.outputs, strict=True):
            replacement[old] = new

    return cast(TensorVar, resolve(output))


def ensure_array(
    value: float | list[float] | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Ensure a value is a numpy array with dtype float64.

    Converts scalars to 0-d arrays and lists to 1-d arrays.
    Existing numpy arrays are converted to float64 dtype if needed.

    Args:
        value: Input value (scalar, list, or array)

    Returns:
        NumPy array with dtype float64
    """
    return np.asarray(value, dtype=np.float64)


def create_bounded_tensor(
    name: str, domain: DomainBounds, kind: Callable[..., TensorVar] = pt.scalar
) -> TensorVar:
    """
    Creates a tensor variable with optional domain constraints.

    Args:
        name: Name of the parameter.
        domain (tuple): Tuple specifying (min, max) range. Use None for unbounded sides.
                       For example: (0.0, None) for lower bound only, (None, 1.0) for upper bound only.
                       If both bounds are None, returns an unbounded tensor.
        kind: pt.scalar for scalars, pt.vector for vectors (default: pt.scalar).

    Returns:
        pytensor.tensor.variable.TensorVariable: The tensor variable, clipped to domain if bounds exist.

    Examples:
        >>> sigma = create_bounded_tensor("sigma", (0.0, None))  # sigma >= 0 (scalar)
        >>> fraction = create_bounded_tensor("fraction", (0.0, 1.0))  # 0 <= fraction <= 1 (scalar)
        >>> temperatures = create_bounded_tensor("temperatures", (None, 100.0), pt.vector)  # vector <= 100
        >>> unbounded = create_bounded_tensor("unbounded", (None, None))  # no bounds applied
    """
    min_bound, max_bound = domain

    # Create the base tensor
    tensor = kind(name)

    # If both bounds are None, return unbounded tensor
    if min_bound is None and max_bound is None:
        return tensor

    # Use infinity constants for unbounded sides
    min_val = pt.constant(-np.inf) if min_bound is None else pt.constant(min_bound)
    max_val = pt.constant(np.inf) if max_bound is None else pt.constant(max_bound)

    clipped = pt.clip(tensor, min_val, max_val)
    clipped.name = tensor.name  # Preserve the original name
    return cast(TensorVar, clipped)
