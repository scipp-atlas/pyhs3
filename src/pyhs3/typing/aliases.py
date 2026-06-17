"""
typing
"""

from __future__ import annotations

from typing import Any, Literal

from pytensor.graph.basic import Apply
from pytensor.tensor.type import TensorType
from pytensor.tensor.variable import TensorVariable

type TensorVar = TensorVariable[TensorType, Apply[Any]]
type EntityType = Literal[
    "distribution", "function", "parameter", "modifier", "constant"
]
type DomainBounds = tuple[float | None, float | None]

__all__ = (
    "DomainBounds",
    "EntityType",
    "TensorVar",
)
