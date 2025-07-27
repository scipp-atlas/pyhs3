"""
typing
"""

from __future__ import annotations

from typing import Any, TypeAlias

from pytensor.graph.basic import Apply
from pytensor.tensor.type import TensorType
from pytensor.tensor.variable import TensorVariable

TensorVar: TypeAlias = TensorVariable[TensorType, Apply[Any]]

__all__ = ("TensorVar",)
