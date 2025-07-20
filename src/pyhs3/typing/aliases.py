"""
typing
"""

from __future__ import annotations

from typing import Any

from pytensor.graph.basic import Apply
from pytensor.tensor.type import TensorType
from pytensor.tensor.variable import TensorVariable

from pyhs3.typing_compat import TypeAlias

TensorVar: TypeAlias = TensorVariable[TensorType, Apply[Any]]

__all__ = ("TensorVar",)
