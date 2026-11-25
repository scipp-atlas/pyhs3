"""
Conversion of pyhf workspaces (HistFactory JSON, "HiFa") to HS3.

:func:`from_hifa` turns a pyhf workspace into a :class:`~pyhs3.Workspace`.
pyhf is not a dependency: the input is any mapping that follows pyhf's
``schemas/1.0.0/workspace.json`` (a :class:`pyhf.Workspace` is a ``dict``
subclass), checked by the private pydantic models below.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from pyhs3.distributions.histfactory.modifiers import (
    HistoSysInterpolation,
    NormSysInterpolation,
)
from pyhs3.workspace import Workspace

__all__ = ["from_hifa"]

HS3_VERSION = "0.2"

NonEmptyFloats = Annotated[list[float], Field(min_length=1)]


class _Schema(BaseModel):
    """Base of the models mirroring pyhf's ``schemas/1.0.0/defs.json``.

    pyhf's schema sets ``additionalProperties: false`` on every object, so an
    unknown key is an error here too.
    """

    model_config = ConfigDict(extra="forbid")


class _NormFactor(_Schema):
    name: str
    type: Literal["normfactor"]
    data: None


class _NormSysData(_Schema):
    hi: float
    lo: float


class _NormSys(_Schema):
    name: str
    type: Literal["normsys"]
    data: _NormSysData


class _HistoSysData(_Schema):
    hi_data: NonEmptyFloats
    lo_data: NonEmptyFloats


class _HistoSys(_Schema):
    name: str
    type: Literal["histosys"]
    data: _HistoSysData


class _ShapeFactor(_Schema):
    name: str
    type: Literal["shapefactor"]
    data: None


class _ShapeSys(_Schema):
    name: str
    type: Literal["shapesys"]
    data: NonEmptyFloats


class _StatError(_Schema):
    name: str
    type: Literal["staterror"]
    data: NonEmptyFloats


class _Lumi(_Schema):
    name: Literal["lumi"]
    type: Literal["lumi"]
    data: None


_Modifier = Annotated[
    _NormFactor | _NormSys | _HistoSys | _ShapeFactor | _ShapeSys | _StatError | _Lumi,
    Field(discriminator="type"),
]


class _Sample(_Schema):
    name: str
    data: NonEmptyFloats
    modifiers: list[_Modifier]


class _Channel(_Schema):
    name: str
    samples: Annotated[list[_Sample], Field(min_length=1)]


class _Observation(_Schema):
    name: str
    data: NonEmptyFloats


class _Parameter(_Schema):
    name: str
    inits: NonEmptyFloats | None = None
    bounds: Annotated[list[tuple[float, float]], Field(min_length=1)] | None = None
    auxdata: NonEmptyFloats | None = None
    factors: NonEmptyFloats | None = None
    sigmas: NonEmptyFloats | None = None
    fixed: bool | None = None


class _Config(_Schema):
    poi: str
    parameters: list[_Parameter]


class _Measurement(_Schema):
    name: str
    config: _Config


class _HiFaWorkspace(_Schema):
    channels: Annotated[list[_Channel], Field(min_length=1)]
    measurements: Annotated[list[_Measurement], Field(min_length=1)]
    observations: Annotated[list[_Observation], Field(min_length=1)]
    version: Literal["1.0.0"]


class _NormSysSettings(_Schema):
    interpcode: NormSysInterpolation


class _HistoSysSettings(_Schema):
    interpcode: HistoSysInterpolation


class _ModifierSettings(_Schema):
    """pyhf's ``modifier_settings``, defaulting to pyhs3's (and ROOT's) codes.

    A modifier type left out keeps pyhs3's default.
    """

    normsys: _NormSysSettings = _NormSysSettings(interpcode="code4")
    histosys: _HistoSysSettings = _HistoSysSettings(interpcode="code4p")


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
# The auxiliary data and widths HS3 v0.2 expresses for a Gauss constraint.
_GAUSS_AUX: dict[str, list[float] | None] = {"auxdata": [0.0], "sigmas": [1.0]}
# lumi's constraint takes its auxiliary data and width from each measurement.
_LUMI_AUX: dict[str, list[float] | None] = {"auxdata": None, "sigmas": None}
# Relative tolerance for config auxdata, factors and sigmas that pyhf derives
# from the sample data, which a writer may have rounded.
_AUX_RTOL = 1e-9


@dataclass(frozen=True)
class _ParameterSet:
    """One pyhf parameter: a scalar, or one entry per bin."""

    modifier_type: str
    defaults: _ParameterDefaults
    # The config ``auxdata``, ``factors`` and ``sigmas`` pyhf derives; a
    # measurement may repeat these but not change them. None: pyhf derives
    # nothing, so each measurement must set it.
    aux: Mapping[str, list[float] | None]
    # Shape (size,): bins with no constraint term in HS3, which pyhf fixes.
    unconstrained: tuple[bool, ...] = (False,)

    @property
    def size(self) -> int:
        """Number of parameters: 1 for a scalar, else the number of bins."""
        return len(self.unconstrained)

    def names(self, name: str) -> list[str]:
        """HS3 names of the parameters of pyhf parameter ``name``."""
        if not self.defaults.per_bin:
            return [name]
        return _bin_names(name, 0, self.size)


@dataclass(frozen=True)
class _StatErrorLayout:
    """Where each pyhf ``staterror`` lives, and how pyhs3 holds it."""

    # name -> channel -> the (sample, modifier) pairs carrying it there.
    groups: Mapping[str, Mapping[str, list[tuple[_Sample, _StatError]]]]
    # name -> channel -> pyhf's global index of that channel's first bin.
    offsets: Mapping[str, Mapping[str, int]]
    # channel -> the one staterror group pyhs3's Barlow-Beeston lite holds.
    lite: Mapping[str, str]


# Where a modifier sits: its channel, its sample and the modifier itself.
_Use = tuple[_Channel, _Sample, _Modifier]


def _bin_names(name: str, start: int, nbins: int) -> list[str]:
    """``name_bin_i`` for ``i`` in ``[start, start + nbins)``.

    ROOT rejects pyhf's ``name[i]`` as an object name.
    """
    return [f"{name}_bin_{i}" for i in range(start, start + nbins)]


def _check_unique(kind: str, names: Iterable[str]) -> None:
    duplicates = sorted(name for name, n in Counter(names).items() if n > 1)
    if duplicates:
        msg = f"Duplicate {kind} names: {duplicates}"
        raise ValueError(msg)


def _check_consistency(hifa: _HiFaWorkspace) -> None:
    """Check names and bin counts, as pyhf does when it builds a model."""
    _check_unique("channel", (c.name for c in hifa.channels))
    _check_unique("observation", (o.name for o in hifa.observations))
    _check_unique("measurement", (m.name for m in hifa.measurements))
    for measurement in hifa.measurements:
        _check_unique(
            f"parameter (measurement '{measurement.name}')",
            (p.name for p in measurement.config.parameters),
        )

    observations = {o.name: o for o in hifa.observations}
    channel_names = {c.name for c in hifa.channels}
    for name in observations.keys() - channel_names:
        msg = f"Observation '{name}' has no channel of that name"
        raise ValueError(msg)

    for channel in hifa.channels:
        nbins = len(channel.samples[0].data)
        for sample in channel.samples:
            if len(sample.data) != nbins:
                msg = (
                    f"Sample '{sample.name}' in channel '{channel.name}' has "
                    f"{len(sample.data)} bins, expected {nbins}"
                )
                raise ValueError(msg)
            for modifier in sample.modifiers:
                if isinstance(modifier, _HistoSys):
                    lengths = {
                        key: len(getattr(modifier.data, key))
                        for key in ("hi_data", "lo_data")
                    }
                elif isinstance(modifier, _ShapeSys | _StatError):
                    lengths = {"data": len(modifier.data)}
                else:
                    continue
                for key, n in lengths.items():
                    if n != nbins:
                        msg = (
                            f"{modifier.type} modifier '{modifier.name}' on sample "
                            f"'{sample.name}' in channel '{channel.name}': {key} "
                            f"has {n} bins, expected {nbins}"
                        )
                        raise ValueError(msg)
        if channel.name not in observations:
            msg = f"channel '{channel.name}' has no observation"
            raise ValueError(msg)
        n_observed = len(observations[channel.name].data)
        if n_observed != nbins:
            msg = (
                f"Observation '{channel.name}' has {n_observed} bins, but channel "
                f"'{channel.name}' has {nbins}"
            )
            raise ValueError(msg)


def _staterror_layout(hifa: _HiFaWorkspace) -> _StatErrorLayout:
    """Group each channel's ``staterror`` modifiers and number their bins as pyhf does.

    pyhf gives a ``staterror`` name one parameter per bin of the channels
    using it, numbered over those channels sorted by name. pyhs3's
    Barlow-Beeston lite holds one staterror group per channel: the group on
    more than one sample if there is one, else the first group to appear.
    Every other group sits on a single sample.

    Raises:
        ValueError: For two staterror groups in one channel that each sit on
            more than one sample, or a sample carrying a staterror name in only
            some of the channels using it (which pyhf rejects).
    """
    groups: dict[str, dict[str, list[tuple[_Sample, _StatError]]]] = {}
    lite: dict[str, str] = {}
    for channel in hifa.channels:
        by_name: dict[str, list[tuple[_Sample, _StatError]]] = {}
        for sample in channel.samples:
            for modifier in sample.modifiers:
                if isinstance(modifier, _StatError):
                    by_name.setdefault(modifier.name, []).append((sample, modifier))
        shared = [name for name, members in by_name.items() if len(members) > 1]
        if len(shared) > 1:
            msg = (
                f"Channel '{channel.name}' has staterror modifiers {shared} each on "
                f"more than one sample; pyhs3's Barlow-Beeston lite holds one "
                f"staterror group per channel"
            )
            raise ValueError(msg)
        if by_name:
            lite[channel.name] = shared[0] if shared else next(iter(by_name))
        for name, members in by_name.items():
            groups.setdefault(name, {})[channel.name] = members

    offsets: dict[str, dict[str, int]] = {}
    for name, channels in groups.items():
        carriers: dict[str, set[str]] = {}
        for channel_name, members in channels.items():
            for sample, _ in members:
                carriers.setdefault(sample.name, set()).add(channel_name)
        for sample_name, carried in carriers.items():
            if carried != channels.keys():
                msg = (
                    f"staterror '{name}' is on sample '{sample_name}' in channels "
                    f"{sorted(carried)}, but is used in channels {sorted(channels)}; "
                    f"pyhf needs each sample carrying a staterror to carry it in "
                    f"every channel using it"
                )
                raise ValueError(msg)
        start = 0
        offsets[name] = {}
        for channel_name in sorted(channels):
            offsets[name][channel_name] = start
            start += len(channels[channel_name][0][0].data)
    return _StatErrorLayout(groups=groups, offsets=offsets, lite=lite)


def _shapefactor_parameter(
    name: str, defaults: _ParameterDefaults, uses: list[_Use]
) -> _ParameterSet:
    """Parameter set of a ``shapefactor``, whose bin ``i`` is shared wherever it appears.

    Raises:
        ValueError: For a shapefactor on channels with different bin counts
            (pyhf raises ``InvalidNameReuse``).
    """
    nbins = {len(sample.data) for _, sample, _ in uses}
    if len(nbins) > 1:
        msg = (
            f"shapefactor '{name}' is used on channels with different bin counts "
            f"{sorted(nbins)}; its parameters are shared bin by bin"
        )
        raise ValueError(msg)
    return _ParameterSet(
        "shapefactor", defaults, aux={}, unconstrained=(False,) * nbins.pop()
    )


def _shapesys_parameter(
    name: str, defaults: _ParameterDefaults, uses: list[_Use]
) -> _ParameterSet:
    """Parameter set of a ``shapesys``, Poisson-constrained with tau = (nu / delta)**2.

    A bin with no positive yield or no positive uncertainty has no tau; pyhf
    fixes it (with tau = 1) and pyhs3 gives it no constraint term.

    Raises:
        ValueError: For a shapesys used on more than one sample or channel,
            whose constraint pyhs3 would emit once per use (pyhf raises
            ``InvalidModel``).
    """
    if len(uses) > 1:
        where = [f"'{sample.name}' in '{channel.name}'" for channel, sample, _ in uses]
        msg = (
            f"shapesys modifier '{name}' is used on more than one sample or channel "
            f"({', '.join(where)}); give each use its own name"
        )
        raise ValueError(msg)
    ((_, sample, modifier),) = uses
    assert isinstance(modifier, _ShapeSys)
    # Shape (nbins,): nominal yields and absolute uncertainties.
    nominal = np.asarray(sample.data, dtype=np.float64)
    delta = np.asarray(modifier.data, dtype=np.float64)
    valid = (nominal > 0) & (delta > 0)
    tau = np.divide(nominal**2, delta**2, out=np.ones_like(nominal), where=valid)
    return _ParameterSet(
        "shapesys",
        defaults,
        aux={"auxdata": tau.tolist(), "factors": tau.tolist()},
        unconstrained=tuple((~valid).tolist()),
    )


def _staterror_parameter(
    name: str, defaults: _ParameterDefaults, layout: _StatErrorLayout
) -> _ParameterSet:
    """Parameter set of a ``staterror``, Gauss-constrained around 1.

    Per bin, pyhf's width is sqrt(sum delta**2) / sum nu over the samples
    carrying the name, the same as pyhs3's Barlow-Beeston lite. A bin whose
    width is 0 (no positive summed yield, or no uncertainty) has none; pyhf
    fixes it (with width 1) and pyhs3 gives it no constraint term.
    """
    channels = layout.groups[name]
    size = sum(len(members[0][0].data) for members in channels.values())
    # Shape (size,): pyhf's relative width of every bin of the parameter.
    width = np.zeros(size)
    for channel_name, members in channels.items():
        # Shape (samples, nbins) -> (nbins,).
        nominal = np.sum([sample.data for sample, _ in members], axis=0)
        delta = np.sqrt(np.sum(np.square([m.data for _, m in members]), axis=0))
        start = layout.offsets[name][channel_name]
        width[start : start + len(nominal)] = np.divide(
            delta, nominal, out=np.zeros_like(delta), where=nominal > 0
        )
    unconstrained = width == 0
    return _ParameterSet(
        "staterror",
        defaults,
        aux={
            "auxdata": [1.0] * size,
            "sigmas": np.where(unconstrained, 1.0, width).tolist(),
        },
        unconstrained=tuple(unconstrained.tolist()),
    )


def _collect_parameters(
    hifa: _HiFaWorkspace, layout: _StatErrorLayout
) -> dict[str, _ParameterSet]:
    """Map each pyhf parameter to its parameter set.

    pyhf correlates modifiers by name. Modifier types whose parameters have
    different defaults cannot share a name (pyhf raises ``InvalidNameReuse``).

    Raises:
        ValueError: For a name shared by modifier types with different defaults,
            or a per-bin modifier pyhs3 cannot express as pyhf builds it.
    """
    types: dict[str, tuple[str, _ParameterDefaults]] = {}
    uses: dict[str, list[_Use]] = {}
    for channel in hifa.channels:
        for sample in channel.samples:
            for modifier in sample.modifiers:
                defaults = _DEFAULTS[modifier.type]
                first_type, first_defaults = types.setdefault(
                    modifier.name, (modifier.type, defaults)
                )
                if first_defaults != defaults:
                    names = " and ".join(sorted({first_type, modifier.type}))
                    msg = (
                        f"Parameter '{modifier.name}' is used by modifier types "
                        f"with different pyhf defaults: {names}"
                    )
                    raise ValueError(msg)
                uses.setdefault(modifier.name, []).append((channel, sample, modifier))

    parameters: dict[str, _ParameterSet] = {}
    for name, (modifier_type, defaults) in types.items():
        if modifier_type == "shapefactor":
            parameters[name] = _shapefactor_parameter(name, defaults, uses[name])
        elif modifier_type == "shapesys":
            parameters[name] = _shapesys_parameter(name, defaults, uses[name])
        elif modifier_type == "staterror":
            parameters[name] = _staterror_parameter(name, defaults, layout)
        elif modifier_type == "lumi":
            parameters[name] = _ParameterSet(modifier_type, defaults, aux=_LUMI_AUX)
        else:
            aux = _GAUSS_AUX if defaults.constraint == "Gauss" else {}
            parameters[name] = _ParameterSet(modifier_type, defaults, aux=aux)
    return parameters


def _emit_modifier(modifier: _Modifier, settings: _ModifierSettings) -> dict[str, Any]:
    """HS3 modifier for a pyhf ``normfactor``, ``normsys``, ``histosys`` or ``lumi``."""
    common = {"name": modifier.name, "type": modifier.type, "parameter": modifier.name}
    if isinstance(modifier, _Lumi):
        # Its constraint is the measurement's ``lumi_constraint_<m>``.
        return {**common, "type": "normfactor"}
    if isinstance(modifier, _NormSys):
        return {
            **common,
            "constraint": "Gauss",
            "data": {
                "hi": modifier.data.hi,
                "lo": modifier.data.lo,
                "interpolation": settings.normsys.interpcode,
            },
        }
    if isinstance(modifier, _HistoSys):
        return {
            **common,
            "constraint": "Gauss",
            "data": {
                "hi": {"contents": list(modifier.data.hi_data)},
                "lo": {"contents": list(modifier.data.lo_data)},
                "interpolation": settings.histosys.interpcode,
            },
        }
    return common


def _emit_per_bin_modifier(
    modifier: _ShapeFactor | _ShapeSys, parameters: list[str]
) -> dict[str, Any]:
    """HS3 modifier for a pyhf ``shapefactor`` or ``shapesys``."""
    common = {"name": modifier.name, "type": modifier.type, "parameters": parameters}
    if isinstance(modifier, _ShapeSys):
        # Absolute uncertainties, as pyhf and pyhs3 (not ROOT) read ``vals``.
        return {
            **common,
            "constraint": "Poisson",
            "data": {"vals": list(modifier.data)},
        }
    return common


def _emit_staterror(
    modifier: _StatError, parameters: list[str], lite: bool
) -> dict[str, Any]:
    """HS3 modifier for a pyhf ``staterror``.

    The channel's lite group is a Barlow-Beeston lite ``staterror``, whose
    uncertainties go in the sample's ``errors``. Any other group sits on a
    single sample, where pyhf's width is |delta| / nominal: exactly a Gauss
    ``shapesys`` with absolute ``vals`` = |delta|.
    """
    common = {"name": modifier.name, "parameters": parameters, "constraint": "Gauss"}
    if lite:
        return {**common, "type": "staterror"}
    # pyhf squares the uncertainties, so their sign does not matter.
    vals = [abs(delta) for delta in modifier.data]
    return {**common, "type": "shapesys", "data": {"vals": vals}}


def _emit_sample(
    sample: _Sample,
    channel: str,
    settings: _ModifierSettings,
    layout: _StatErrorLayout,
) -> dict[str, Any]:
    """HS3 sample of one pyhf sample in channel ``channel``."""
    nbins = len(sample.data)
    data: dict[str, list[float]] = {"contents": list(sample.data)}
    modifiers: list[dict[str, Any]] = []
    for modifier in sample.modifiers:
        if isinstance(modifier, _StatError):
            start = layout.offsets[modifier.name][channel]
            lite = layout.lite[channel] == modifier.name
            if lite:
                data["errors"] = list(modifier.data)
            modifiers.append(
                _emit_staterror(modifier, _bin_names(modifier.name, start, nbins), lite)
            )
        elif isinstance(modifier, _ShapeFactor | _ShapeSys):
            parameters = _bin_names(modifier.name, 0, nbins)
            modifiers.append(_emit_per_bin_modifier(modifier, parameters))
        else:
            modifiers.append(_emit_modifier(modifier, settings))
    return {"name": sample.name, "data": data, "modifiers": modifiers}


def _emit_channel(
    channel: _Channel,
    observation: _Observation,
    settings: _ModifierSettings,
    layout: _StatErrorLayout,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """HS3 ``histfactory_dist`` and ``binned`` data for one pyhf channel."""
    nbins = len(channel.samples[0].data)
    # hist2workspace's convention: one unit-width bin per pyhf bin.
    axis = {"name": f"obs_x_{channel.name}", "min": 0, "max": nbins, "nbins": nbins}
    distribution = {
        "type": "histfactory_dist",
        "name": f"model_{channel.name}",
        "axes": [axis],
        "samples": [
            _emit_sample(sample, channel.name, settings, layout)
            for sample in channel.samples
        ],
    }
    datum = {
        "type": "binned",
        "name": f"obsData_{channel.name}",
        "axes": [dict(axis)],
        "contents": list(observation.data),
    }
    return distribution, datum


def _resolve_parameter(
    measurement: str,
    name: str,
    parameter: _ParameterSet,
    entry: _Parameter | None,
) -> list[tuple[float, tuple[float, float], bool]]:
    """Initial value, bounds and fixed flag of each entry of one parameter.

    A bin with no constraint term is fixed whatever the config says, so an
    entry setting ``fixed`` to false for such a parameter raises.

    Raises:
        ValueError: For a config entry pyhf would reject or HS3 v0.2 cannot
            express, or a missing one pyhf has no default for.
    """
    defaults = parameter.defaults
    size = parameter.size
    where = f"Measurement '{measurement}': parameter '{name}'"
    # The config keys pyhf has no default for; pyhf fails without them.
    required = [
        key
        for key, default in [
            ("inits", defaults.init),
            ("bounds", defaults.bounds),
            *parameter.aux.items(),
        ]
        if default is None
    ]
    missing = [k for k in required if entry is None or getattr(entry, k) is None]
    if missing:
        msg = (
            f"{where} is a {parameter.modifier_type}, which needs {missing} in "
            f"the measurement config"
        )
        raise ValueError(msg)
    if entry is None:
        entry = _Parameter(name=name)
    for key in ("inits", "bounds", *required):
        values = getattr(entry, key)
        if values is not None and len(values) != size:
            msg = f"{where} has {len(values)} {key}, expected {size}"
            raise ValueError(msg)
    for key in ("auxdata", "sigmas", "factors"):
        values = getattr(entry, key)
        if values is None:
            continue
        if key not in parameter.aux:
            msg = f"{where} is a {parameter.modifier_type}, which does not use {key}"
            raise ValueError(msg)
        expected = parameter.aux[key]
        if expected is None:
            if key == "sigmas" and min(values) <= 0:
                # pyhf evaluates the constraint to NaN instead of raising.
                msg = f"{where} sets sigmas={list(values)}, which must be positive"
                raise ValueError(msg)
            continue
        if len(values) != len(expected) or not np.allclose(
            values, expected, rtol=_AUX_RTOL, atol=0.0
        ):
            msg = (
                f"{where} sets {key}={list(values)}; HS3 v0.2 only expresses "
                f"pyhf's default {key}={expected}"
            )
            raise ValueError(msg)
    if entry.fixed is False and any(parameter.unconstrained):
        bins = [i for i, u in enumerate(parameter.unconstrained) if u]
        msg = (
            f"{where} sets fixed=false, but bins {bins} have no constraint term "
            f"in HS3 v0.2; pyhf would free them under a dummy constraint"
        )
        raise ValueError(msg)
    # The check of ``required`` above leaves no None default in use here.
    inits = cast(
        list[float], entry.inits if entry.inits is not None else [defaults.init] * size
    )
    bounds = cast(
        list[tuple[float, float]],
        entry.bounds if entry.bounds is not None else [defaults.bounds] * size,
    )
    return [
        (init, bound, bool(entry.fixed) or unconstrained)
        for init, bound, unconstrained in zip(
            inits, bounds, parameter.unconstrained, strict=True
        )
    ]


def _emit_measurement(
    measurement: _Measurement,
    parameters: Mapping[str, _ParameterSet],
    distributions: list[str],
    data: list[str],
) -> dict[str, Any]:
    """HS3 likelihood, analysis, domain, parameter point and constraints of one measurement.

    ``lumi`` gets a Gauss constraint ``lumi_constraint_<m>`` in the likelihood's
    ``aux_distributions``, built from this measurement's ``auxdata`` and
    ``sigmas``. A fixed ``lumi`` keeps it: pyhf still evaluates the
    constraint at the fixed value, so the likelihoods agree.
    """
    name = measurement.name
    # Entries for parameters absent from the model are ignored, as in pyhf.
    entries = {p.name: p for p in measurement.config.parameters}
    points: list[dict[str, Any]] = []
    axes: list[dict[str, Any]] = []
    for parameter in sorted(parameters):
        parameter_set = parameters[parameter]
        resolved = _resolve_parameter(
            name, parameter, parameter_set, entries.get(parameter)
        )
        for hs3_name, (init, (lo, hi), fixed) in zip(
            parameter_set.names(parameter), resolved, strict=True
        ):
            points.append({"name": hs3_name, "value": init, "const": fixed})
            axes.append({"name": hs3_name, "min": lo, "max": hi})
    likelihood: dict[str, Any] = {
        "name": name,
        "distributions": distributions,
        "data": data,
    }
    constraints: list[dict[str, Any]] = []
    if "lumi" in parameters:
        lumi = entries["lumi"]
        # _resolve_parameter checked that the config sets both, one value each.
        assert lumi.auxdata is not None
        assert lumi.sigmas is not None
        constraints.append(
            {
                "type": "gaussian_dist",
                "name": f"lumi_constraint_{name}",
                "x": lumi.auxdata[0],
                "mean": "lumi",
                "sigma": lumi.sigmas[0],
            }
        )
        likelihood["aux_distributions"] = [c["name"] for c in constraints]
    return {
        "likelihood": likelihood,
        "constraints": constraints,
        "analysis": {
            "name": name,
            "likelihood": name,
            "parameters_of_interest": [measurement.config.poi],
            "domains": [name],
            "init": name,
        },
        "domain": {"name": name, "type": "product_domain", "axes": axes},
        "parameter_point": {"name": name, "parameters": points},
    }


def from_hifa(
    spec: Mapping[str, Any],
    *,
    modifier_settings: Mapping[str, Mapping[str, str]] | None = None,
) -> Workspace:
    """Convert a pyhf workspace (HistFactory JSON) to an HS3 workspace.

    The input is a workspace following pyhf's ``schemas/1.0.0/workspace.json``,
    with any patches already applied (a :class:`pyhf.Workspace` works, since it
    is a ``dict``). The result goes through all of :class:`~pyhs3.Workspace`'s
    validation.

    Every pyhf modifier type is supported: ``normfactor``, ``normsys``,
    ``histosys``, ``shapefactor``, ``shapesys``, ``staterror`` and ``lumi``.

    **Names of the generated HS3 objects**

    ============================  =================================================
    pyhf                          HS3
    ============================  =================================================
    channel ``c`` with ``n`` bins ``histfactory_dist`` ``model_c`` with axis
                                  ``obs_x_c`` (``min=0``, ``max=n``, ``nbins=n``)
    observation of channel ``c``  ``binned`` data ``obsData_c`` on axis ``obs_x_c``
    measurement ``m``             likelihood, analysis, ``product_domain`` and
                                  parameter point, all named ``m``
    scalar modifier ``x``         parameter ``x`` (no ``alpha_`` prefix)
    per-bin modifier ``x``        parameters ``x_bin_0``, ``x_bin_1``, ... (pyhf's
                                  ``x[i]``, which ROOT rejects as a name)
    ``lumi`` in measurement ``m`` ``gaussian_dist`` ``lumi_constraint_m``, in
                                  likelihood ``m``'s ``aux_distributions``
    ============================  =================================================

    A ``shapefactor`` or ``shapesys`` numbers the bins of its own channel. A
    ``staterror`` numbers them as pyhf does: over the channels using its name,
    sorted by channel name, so ``x_bin_i`` is pyhf's ``x[i]``. Its parameters
    stay per channel.

    Each likelihood pairs every channel's distribution with its data, in the
    input's channel order. Each analysis has ``init`` and its one domain set to
    ``m``, and ``parameters_of_interest`` set to the measurement's ``poi``,
    copied as given even when no modifier uses it. Parameter point ``m`` and
    domain ``m`` list every parameter of the model, sorted by pyhf name, with
    per-bin parameters in bin order.

    **Parameter defaults** (pyhf's), which the measurement config's ``inits``,
    ``bounds`` and ``fixed`` override; a fixed parameter is ``const``:

    ===========================  ====  ===========  ==============================
    modifier                     init  bounds       constraint
    ===========================  ====  ===========  ==============================
    ``normfactor``               1     [0, 10]      none
    ``normsys``, ``histosys``    0     [-5, 5]      ``Gauss`` (aux 0, width 1)
    ``shapefactor``              1     [0, 10]      none
    ``shapesys``                 1     [1e-10, 10]  ``Poisson``, tau = (nu/delta)^2
    ``staterror``                1     [1e-10, 10]  ``Gauss``, width
                                                    sqrt(sum delta^2) / sum nu
    ``lumi``                     none  none         ``Gauss``, config ``auxdata``
                                                    and ``sigmas``
    ===========================  ====  ===========  ==============================

    For a per-bin parameter, ``inits`` and ``bounds`` hold one entry per bin
    and ``fixed`` applies to every bin. Config entries naming a parameter the
    model does not use are ignored, as pyhf ignores them.

    **Per-bin modifiers.** A ``shapefactor`` maps to HS3 ``shapefactor``, and
    bin ``i`` is shared wherever its name appears, as in pyhf. A ``shapesys``
    maps to HS3 ``shapesys`` with ``constraint: Poisson`` and its absolute
    uncertainties as ``vals``. A ``staterror`` maps to Barlow-Beeston lite: an
    HS3 ``staterror`` with ``constraint: Gauss`` on each sample carrying it,
    whose uncertainties go in that sample's ``data.errors``; samples without it
    get no ``errors``. pyhs3's lite mode holds one staterror group per channel:
    the group on more than one sample if there is one, else the first to
    appear. Any other group in the channel sits on a single sample and maps to
    a ``shapesys`` with ``constraint: Gauss`` and ``vals`` = |delta|, which
    gives the same width delta / nu.

    A ``shapesys`` bin with no positive yield or no positive uncertainty, and a
    ``staterror`` bin with no positive summed yield or a combined uncertainty
    of 0, has no constraint term in pyhs3. pyhf fixes such a bin, so its
    parameter is ``const`` at its initial value (1 unless ``inits`` says
    otherwise).

    **lumi.** pyhf requires ``inits``, ``bounds``, ``auxdata`` and ``sigmas``
    (one value each) for ``lumi`` in every measurement. ``lumi`` maps to a
    ``normfactor`` on parameter ``lumi`` plus, per measurement ``m``, a
    ``gaussian_dist`` with ``x`` = auxdata, ``mean`` = ``lumi`` and ``sigma``
    = sigmas. A fixed ``lumi`` is ``const`` and keeps its constraint, which
    pyhf still evaluates at the fixed value.

    **Interpolation.** Each ``normsys`` and ``histosys`` modifier records its
    code in ``data.interpolation``. The defaults are pyhs3's (and ROOT's):
    ``code4`` for ``normsys`` and ``code4p`` for ``histosys``.
    ``modifier_settings`` overrides them per modifier type, in pyhf's shape,
    e.g. ``{"normsys": {"interpcode": "code1"}}``; a type it leaves out keeps
    pyhs3's default. Accepted codes are ``code1`` and ``code4`` for ``normsys``,
    and ``code0``, ``code2`` and ``code4p`` for ``histosys``.

    Args:
        spec: The pyhf workspace.
        modifier_settings: Interpolation code per modifier type, as passed to
            ``pyhf.Workspace.model(modifier_settings=...)``.

    Returns:
        The HS3 workspace.

    Raises:
        pydantic.ValidationError: If ``spec`` does not follow pyhf's schema, or
            ``modifier_settings`` names an unknown modifier type, key or code.
        ValueError: For duplicate channel, observation, measurement or
            config-parameter names; a channel without an observation or an
            observation without a channel; bin counts that disagree within a
            channel, including a ``histosys``, ``shapesys`` or ``staterror``'s
            data; one name used by modifier types with different pyhf
            defaults, such as ``normfactor`` with ``normsys`` or ``shapefactor``,
            ``shapesys`` with ``staterror``, and ``lumi`` with any other type; a
            ``shapefactor`` on channels with different bin counts; a ``shapesys`` name used on more than one
            sample or channel, whose constraint pyhs3 would emit once per use;
            two ``staterror`` groups in one channel that each sit on more than
            one sample; a sample carrying a ``staterror`` in only some of the
            channels using its name, which pyhf rejects; config ``inits`` or
            ``bounds`` whose length is not the parameter's number of bins (1 for
            a scalar); config ``fixed: false`` on a per-bin parameter with a bin
            that has no constraint term, which pyhf would free under a dummy
            constraint; config ``auxdata``, ``sigmas`` or ``factors`` a
            modifier type does not use (all three on ``normfactor`` and
            ``shapefactor``, ``factors`` on ``normsys``, ``histosys`` and
            ``staterror``, ``sigmas`` on ``shapesys``); and config ``auxdata``,
            ``sigmas`` or ``factors`` other than the values pyhf derives,
            within a relative tolerance of 1e-9, which HS3 v0.2 cannot
            express: ``[0]`` and ``[1]`` for ``normsys``/``histosys``, tau
            (1 for a bin with no constraint term) for ``shapesys``, and
            auxdata 1 and the width (1 for a bin with no constraint term) for
            ``staterror``; a ``lumi`` whose config lacks ``inits``,
            ``bounds``, ``auxdata`` or ``sigmas``, sets more than one value
            for any of them, sets ``factors``, or sets a non-positive
            ``sigmas``.

    Examples:
        >>> import pyhs3
        >>> spec = {
        ...     "channels": [
        ...         {
        ...             "name": "SR",
        ...             "samples": [
        ...                 {
        ...                     "name": "signal",
        ...                     "data": [5.0, 10.0],
        ...                     "modifiers": [
        ...                         {"name": "mu", "type": "normfactor", "data": None}
        ...                     ],
        ...                 },
        ...                 {
        ...                     "name": "background",
        ...                     "data": [50.0, 60.0],
        ...                     "modifiers": [
        ...                         {
        ...                             "name": "bkg_norm",
        ...                             "type": "normsys",
        ...                             "data": {"hi": 1.1, "lo": 0.9},
        ...                         }
        ...                     ],
        ...                 },
        ...             ],
        ...         }
        ...     ],
        ...     "observations": [{"name": "SR", "data": [53.0, 65.0]}],
        ...     "measurements": [
        ...         {"name": "meas", "config": {"poi": "mu", "parameters": []}}
        ...     ],
        ...     "version": "1.0.0",
        ... }
        >>> ws = pyhs3.from_hifa(spec)
        >>> [p.name for p in ws.parameter_points["meas"]]
        ['bkg_norm', 'mu']
        >>> ws.domains["meas"]["mu"]
        (0.0, 10.0)
    """
    hifa = _HiFaWorkspace.model_validate(dict(spec))
    settings = _ModifierSettings.model_validate(dict(modifier_settings or {}))
    _check_consistency(hifa)
    layout = _staterror_layout(hifa)
    parameters = _collect_parameters(hifa, layout)

    observations = {o.name: o for o in hifa.observations}
    distributions: list[dict[str, Any]] = []
    data: list[dict[str, Any]] = []
    for channel in hifa.channels:
        distribution, datum = _emit_channel(
            channel, observations[channel.name], settings, layout
        )
        distributions.append(distribution)
        data.append(datum)

    measurements = [
        _emit_measurement(
            measurement,
            parameters,
            [d["name"] for d in distributions],
            [d["name"] for d in data],
        )
        for measurement in hifa.measurements
    ]
    hs3: dict[str, Any] = {
        "metadata": {"hs3_version": HS3_VERSION},
        "distributions": distributions
        + [c for m in measurements for c in m["constraints"]],
        "data": data,
        "likelihoods": [m["likelihood"] for m in measurements],
        "analyses": [m["analysis"] for m in measurements],
        "domains": [m["domain"] for m in measurements],
        "parameter_points": [m["parameter_point"] for m in measurements],
    }
    return Workspace(**hs3)
