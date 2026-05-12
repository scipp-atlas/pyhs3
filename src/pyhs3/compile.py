"""
Compatibility shim for pytensor's compile.function across pytensor 2 and 3.

pytensor 2: pytensor.compile.function is a submodule; function lives at
            pytensor.compile.function.function
pytensor 3: the submodule was removed; function lives at pytensor.compile.function
"""

from __future__ import annotations

try:
    # pytensor 3
    from pytensor.compile.maker import function
except ImportError:
    # pytensor 2
    from pytensor.compile.function import (  # type: ignore[no-redef]
        function,
    )

__all__ = ["function"]
