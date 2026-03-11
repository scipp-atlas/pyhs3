"""
Copyright (c) 2025 Giordon Stark. All rights reserved.

pyhs3: pure-Python HS3 implementation with tensors and autodiff
"""

from __future__ import annotations

from pyhs3._version import version as __version__
from pyhs3.logging import setup
from pyhs3.model import Model
from pyhs3.workspace import Workspace

setup()

__all__ = [
    "Model",
    "Workspace",
    "__version__",
]
