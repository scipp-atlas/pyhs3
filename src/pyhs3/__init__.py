"""
Copyright (c) 2025 Giordon Stark. All rights reserved.

pyhs3: pure-Python HS3 implementation with tensors and autodiff
"""

from __future__ import annotations

import pytensor

from pyhs3._version import version as __version__
from pyhs3.logging import setup
from pyhs3.model import Model
from pyhs3.transpile import JaxifiedGraph, jaxify
from pyhs3.workspace import Workspace

setup()

# keep cvm linker for now: https://github.com/pymc-devs/pytensor/pull/1862
pytensor.config.linker = "cvm" if pytensor.config.cxx else "vm"  # type: ignore[attr-defined]

__all__ = [
    "JaxifiedGraph",
    "Model",
    "Workspace",
    "__version__",
    "jaxify",
]
