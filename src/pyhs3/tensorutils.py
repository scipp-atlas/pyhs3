from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt
import pytensor.tensor as pt

from pyhs3.typing.aliases import DomainBounds, TensorVar


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
