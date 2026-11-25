"""
Copyright (c) 2025 Giordon Stark. All rights reserved.

pyhs3: pure-Python HS3 implementation with tensors and autodiff
"""

from __future__ import annotations

from pyhs3._version import version as __version__
from pyhs3.compiled import CompiledNLL
from pyhs3.hifa import from_hifa, hifa_modifier_settings, to_hifa
from pyhs3.model import Model
from pyhs3.transpile import JaxifiedGraph, jaxify
from pyhs3.workspace import Workspace

__all__ = [
    "CompiledNLL",
    "JaxifiedGraph",
    "Model",
    "Workspace",
    "__version__",
    "from_hifa",
    "hifa_modifier_settings",
    "jaxify",
    "to_hifa",
]
