"""
pyhf's parameter defaults and the naming of per-bin parameters.

Both directions of the pyhf <-> HS3 conversion use these.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class _ParameterDefaults:
    """pyhf's defaults for the parameter of one modifier type."""

    # None: pyhf has no default, so the measurement config must set it.
    init: float | None
    bounds: tuple[float, float] | None
    # The constraint pyhf puts on the parameter, if any.
    constraint: Literal["Gauss", "Poisson"] | None
    # One parameter per bin (``name_bin_i``) instead of a scalar.
    per_bin: bool = False


_NORMFACTOR_DEFAULTS = _ParameterDefaults(init=1.0, bounds=(0.0, 10.0), constraint=None)
_NORMAL_DEFAULTS = _ParameterDefaults(init=0.0, bounds=(-5.0, 5.0), constraint="Gauss")
_DEFAULTS: dict[str, _ParameterDefaults] = {
    "normfactor": _NORMFACTOR_DEFAULTS,
    "normsys": _NORMAL_DEFAULTS,
    "histosys": _NORMAL_DEFAULTS,
    "shapefactor": _ParameterDefaults(
        init=1.0, bounds=(0.0, 10.0), constraint=None, per_bin=True
    ),
    "shapesys": _ParameterDefaults(
        init=1.0, bounds=(1e-10, 10.0), constraint="Poisson", per_bin=True
    ),
    "staterror": _ParameterDefaults(
        init=1.0, bounds=(1e-10, 10.0), constraint="Gauss", per_bin=True
    ),
    "lumi": _ParameterDefaults(init=None, bounds=None, constraint="Gauss"),
}


def _bin_names(name: str, start: int, nbins: int) -> list[str]:
    """``name_bin_i`` for ``i`` in ``[start, start + nbins)``.

    ROOT rejects pyhf's ``name[i]`` as an object name.
    """
    return [f"{name}_bin_{i}" for i in range(start, start + nbins)]
