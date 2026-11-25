"""
The pyhf parameters an HS3 workspace converts to, for :func:`~pyhs3.to_hifa`.

pyhf correlates modifiers by name and HS3 by parameter. The registry here
checks that each HS3 parameter becomes exactly one pyhf parameter, or one bin
of one, numbered as pyhf numbers it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pyhs3.hifa._defaults import _DEFAULTS

# The HS3 names of pyhf's per-bin parameter ``name[i]``, tried in order:
# ``name_bin_i`` (from_hifa's), ``name_bin<i>`` (pyhs3's default staterror
# parameters ``staterror_<channel>_bin<i>``) and ``name_i`` (ROOT's). The order
# matters: ``x_bin_0`` also reads as ``name_i`` with name ``x_bin``.
_BIN_NAMES = (
    re.compile(r"(?P<name>.+)_bin_(?P<index>0|[1-9]\d*)"),
    re.compile(r"(?P<name>.+)_bin(?P<index>0|[1-9]\d*)"),
    re.compile(r"(?P<name>.+)_(?P<index>0|[1-9]\d*)"),
)

# pyhf's types of per-bin parameters, as this converter emits them.
_PerBinType = Literal["shapefactor", "shapesys", "staterror"]


@dataclass
class _PerBinUse:
    """Where one pyhf per-bin parameter sits in one channel."""

    start: int
    # The HS3 parameters of its bins here, in bin order.
    parameters: list[str]
    # The HS3 modifiers carrying it there, for the error messages.
    modifiers: list[str] = field(default_factory=list)
    # A Barlow-Beeston lite group, which may span several samples.
    lite: bool = False

    @property
    def nbins(self) -> int:
        return len(self.parameters)


@dataclass
class _PyhfParameter:
    """One pyhf parameter (a modifier name) and the HS3 parameters behind it."""

    modifier_type: str
    # channel -> where a per-bin parameter sits in it; empty for a scalar.
    uses: dict[str, _PerBinUse] = field(default_factory=dict)
    # HS3 parameters of the bins pyhf fixes on its own: no yield or no
    # uncertainty, and so no constraint term in pyhs3 either.
    auto_fixed: set[str] = field(default_factory=set)

    @property
    def per_bin(self) -> bool:
        return _DEFAULTS[self.modifier_type].per_bin

    def hs3_names(self, name: str) -> list[str]:
        """HS3 parameters of pyhf parameter ``name``, in pyhf's bin order."""
        if not self.per_bin:
            return [name]
        if self.modifier_type == "staterror":
            # pyhf numbers a staterror over the channels using it, by name.
            return [
                hs3_name
                for channel in sorted(self.uses)
                for hs3_name in self.uses[channel].parameters
            ]
        return next(iter(self.uses.values())).parameters


def _bin_group(parameters: list[str]) -> tuple[str, int] | None:
    """The pyhf name and first bin of one per-bin group of HS3 parameters.

    The parameters are one group when, for one of the accepted naming
    patterns, all share one name and their indices count up by one from the
    first; None otherwise.
    """
    for pattern in _BIN_NAMES:
        matches = [m for p in parameters if (m := pattern.fullmatch(p))]
        if not parameters or len(matches) != len(parameters):
            continue
        name, start = matches[0]["name"], int(matches[0]["index"])
        if all(
            m["name"] == name and int(m["index"]) == start + offset
            for offset, m in enumerate(matches)
        ):
            return name, start
    return None


def _format(name: str, index: int | None) -> str:
    """pyhf's name of a scalar parameter, or of bin ``index`` of a per-bin one."""
    return name if index is None else f"{name}[{index}]"


class _Registry:
    """The pyhf parameters of the converted channels.

    pyhf correlates modifiers by name, HS3 by parameter, so each HS3 parameter
    must become exactly one pyhf parameter (or one bin of one).
    """

    def __init__(self, functions: set[str]) -> None:
        self.parameters: dict[str, _PyhfParameter] = {}
        self._functions = functions
        # HS3 parameter -> the pyhf parameter (and bin) it becomes.
        self._owners: dict[str, tuple[str, int | None]] = {}
        # pyhf parameter (and bin) -> the HS3 parameter behind it.
        self._sources: dict[tuple[str, int | None], str] = {}

    def _claim(self, hs3_name: str, name: str, index: int | None) -> None:
        if hs3_name in self._functions:
            msg = (
                f"Parameter '{hs3_name}' is a function in the workspace; pyhf "
                f"parameters are free parameters"
            )
            raise ValueError(msg)
        owner = self._owners.setdefault(hs3_name, (name, index))
        if owner != (name, index):
            msg = (
                f"HS3 parameter '{hs3_name}' would be both pyhf's "
                f"{_format(*owner)} and {_format(name, index)}"
            )
            raise ValueError(msg)
        source = self._sources.setdefault((name, index), hs3_name)
        if source != hs3_name:
            msg = (
                f"pyhf's {_format(name, index)} would be both HS3 parameter "
                f"'{source}' and '{hs3_name}'"
            )
            raise ValueError(msg)

    def _parameter(self, modifier_type: str, name: str) -> _PyhfParameter:
        parameter = self.parameters.setdefault(name, _PyhfParameter(modifier_type))
        if _DEFAULTS[parameter.modifier_type] != _DEFAULTS[modifier_type]:
            names = " and ".join(sorted({parameter.modifier_type, modifier_type}))
            msg = (
                f"Parameter '{name}' is used by modifier types with different "
                f"pyhf defaults: {names}"
            )
            raise ValueError(msg)
        return parameter

    def add_scalar(self, modifier_type: str, name: str) -> str:
        """Register a scalar parameter; return its pyhf name."""
        self._parameter(modifier_type, name)
        self._claim(name, name, None)
        return name

    def add_per_bin(
        self,
        modifier_type: _PerBinType,
        modifier: str,
        parameters: list[str],
        channel: str,
        *,
        lite: bool = False,
    ) -> str:
        """Register the parameters of one per-bin modifier; return the pyhf name.

        Raises:
            ValueError: For parameters that are not one group ``name_bin_i``,
                ``name_bin<i>`` or ``name_i`` with consecutive ``i``, or a
                modifier pyhf cannot share where HS3 shares it.
        """
        group = _bin_group(parameters)
        if group is None:
            msg = (
                f"{modifier_type} '{modifier}' in channel '{channel}' has parameters "
                f"{parameters}, which are not one group <name>_bin_<i>, "
                f"<name>_bin<i> or <name>_<i> with consecutive i; pyhf names the "
                f"bins of one modifier name[0], name[1], ..."
            )
            raise ValueError(msg)
        name, start = group
        parameter = self._parameter(modifier_type, name)
        for index, hs3_name in enumerate(parameters, start=start):
            self._claim(hs3_name, name, index)
        use = parameter.uses.setdefault(channel, _PerBinUse(start, list(parameters)))
        if use.parameters != list(parameters):
            msg = (
                f"{modifier_type} '{name}' has parameter lists "
                f"{use.parameters} and {parameters} in channel '{channel}'; pyhf "
                f"gives every use of one name in a channel the same bins"
            )
            raise ValueError(msg)
        if use.modifiers and not (lite and use.lite):
            # Only a lite group spans samples; pyhf would combine any other
            # per-bin modifier's uses into one constraint, which pyhs3 does not.
            msg = (
                f"{modifier_type} '{modifier}' shares its parameters with "
                f"{use.modifiers} in channel '{channel}': pyhf combines the uses "
                f"of one name on more than one sample into one constraint, but "
                f"pyhs3 constrains each shapesys on its own"
            )
            raise ValueError(msg)
        use.modifiers.append(modifier)
        use.lite = lite
        return name

    def check(self) -> None:
        """Check that pyhf numbers and shares each per-bin parameter as HS3 does.

        Raises:
            ValueError: For a shapefactor or shapesys whose bins do not start
                at 0, a shapefactor with different bin counts in different
                channels (whose parameter lists then partly overlap), or a
                staterror whose bins are not pyhf's global index.
        """
        for name, parameter in self.parameters.items():
            if not parameter.per_bin:
                continue
            # A shapesys starting at bin 0 in two channels would share its
            # parameters across them, which the workspace already rejects.
            channels = sorted(parameter.uses)
            nbins = {parameter.uses[c].nbins for c in channels}
            if parameter.modifier_type == "shapefactor" and len(nbins) > 1:
                msg = (
                    f"shapefactor '{name}' has {sorted(nbins)} bins in channels "
                    f"{channels}, so its parameter lists partly overlap; pyhf "
                    f"shares shapefactor bins only between channels with the "
                    f"same bin count"
                )
                raise ValueError(msg)
            expected = 0
            for channel in channels:
                use = parameter.uses[channel]
                if use.start != expected:
                    msg = (
                        f"{parameter.modifier_type} '{name}' in channel "
                        f"'{channel}' starts at bin {use.start}, but pyhf "
                        f"numbers its bins from {expected}"
                    )
                    raise ValueError(msg)
                if parameter.modifier_type == "staterror":
                    # Global index over the channels using it, sorted by name.
                    expected += use.nbins
