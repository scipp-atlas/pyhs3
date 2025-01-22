"""
Copyright (c) 2025 Giordon Stark. All rights reserved.

pyhs3: pure-Python HS3 implementation with tensors and autodiff
"""

from __future__ import annotations

from typing import Any

from ._version import version as __version__


class Workspace:
    """
    HS3 Workspace Object
    """

    def __init__(self, _spec: dict[str, Any]): ...
    def model(self) -> Model:
        return Model()

    def data(self) -> list[float]:
        return [0.0, 0.0, 0.0, 0.0]


class Model:
    """
    HS3 Model Object
    """

    def pdf(self, _pars: list[float], _data: list[float]) -> float:
        return 0.0

    def logpdf(self, _pars: list[float], _data: list[float]) -> float:
        return 0.0


__all__ = ["__version__", "Workspace"]
