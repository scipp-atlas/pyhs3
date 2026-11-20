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

    Separately built HS3 objects often produce the same subexpression, for
    example the same parameter broadcast against the observable in several
    distributions, or the same ``x - m0`` inside different functions.
    :meth:`pyhs3.Model.log_prob` passes its graph through this function so
    that PyTensor receives each subexpression once.  Distributions that are
    identical as HS3 objects are already shared while the model is built
    (see :meth:`pyhs3.distributions.Distribution.structural_key`); this pass
    catches the duplicates below that level.

    Two apply nodes are duplicates when they apply an equal ``Op`` to the
    same inputs, after those inputs have themselves been merged.  Constants
    are the same input when their type, shape and value are equal.  The
    sweep visits each node once in topological order, so one pass merges
    whole duplicate subgraphs, bottom up.

    Not duplicates, and kept as they are:

    - expressions that are only mathematically equal (``x * y`` and
      ``y * x``, or ``exp(log(x))`` and ``x``): no algebra is applied;
      PyTensor's canonicalisation rewrites handle those during compilation;
    - constants that differ in dtype (``2.0`` as float64 vs float32) or
      shape (``[2.0]`` vs ``2.0``), even if their values print the same;
    - distinct graph inputs, even with the same name or the same value at
      evaluation time.

    PyTensor's own merge rewrite finds the same duplicates during
    compilation, but it runs inside the rewriter, replacing one node at a
    time on a ``FunctionGraph`` that validates and notifies its features on
    every replacement.  On the 14-channel bbyy workspace that pass took
    321 s for 404k nodes; this sweep finds the same 148,786 unique nodes in
    about 2.5 s.

    Graph inputs (parameters and observables) are kept as the same variable
    objects, so names and :func:`~pytensor.graph.traversal.explicit_graph_inputs`
    are unchanged.

    Examples:
        The same expression built twice becomes one subgraph:

        >>> import pytensor.tensor as pt
        >>> from pytensor.graph.traversal import io_toposort
        >>> x, y = pt.dscalars("x", "y")
        >>> output = pt.exp(x * y) + pt.exp(x * y)
        >>> len(io_toposort([], [output]))  # 2 Mul, 2 Exp, 1 Add
        5
        >>> len(io_toposort([], [merge_duplicate_nodes(output)]))  # 1 Mul, 1 Exp, 1 Add
        3

        Operands in a different order are not a duplicate:

        >>> output = pt.exp(x * y) + pt.exp(y * x)
        >>> merge_duplicate_nodes(output) is output
        True

        Equal constants are one input, constants of another dtype are not:

        >>> v = pt.dvector("v")
        >>> same = (v + pt.constant(2.0)) * (v + pt.constant(2.0))
        >>> len(io_toposort([], [same]))  # 2 broadcasts of 2.0, 2 Add, 1 Mul
        5
        >>> len(io_toposort([], [merge_duplicate_nodes(same)]))  # 1, 1, 1
        3
        >>> two64 = pt.constant(2.0, dtype="float64")
        >>> two32 = pt.constant(2.0, dtype="float32")
        >>> mixed = (v + two64) * (v + two32)
        >>> merge_duplicate_nodes(mixed) is mixed
        True

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
