"""
typing
"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pytensor.graph.basic import Apply
from pytensor.tensor.type import TensorType
from pytensor.tensor.variable import TensorVariable

TensorVar: TypeAlias = TensorVariable[TensorType, Apply[Any]]
EntityType: TypeAlias = Literal[
    "distribution", "function", "parameter", "modifier", "constant"
]

__all__ = (
    "EntityType",
    "TensorVar",
)
