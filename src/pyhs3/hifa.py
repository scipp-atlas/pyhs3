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
from typing import Annotated, Any, Literal

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

    init: float
    bounds: tuple[float, float]
    # A Gauss-constrained parameter, whose auxiliary datum is 0 and width 1.
    constrained: bool


_NORMFACTOR_DEFAULTS = _ParameterDefaults(
    init=1.0, bounds=(0.0, 10.0), constrained=False
)
_NORMAL_DEFAULTS = _ParameterDefaults(init=0.0, bounds=(-5.0, 5.0), constrained=True)
_DEFAULTS: dict[str, _ParameterDefaults] = {
    "normfactor": _NORMFACTOR_DEFAULTS,
    "normsys": _NORMAL_DEFAULTS,
    "histosys": _NORMAL_DEFAULTS,
}
# The auxiliary data and widths HS3 v0.2 expresses for a Gauss constraint.
_GAUSS_AUX = {"auxdata": [0.0], "sigmas": [1.0]}


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
                if not isinstance(modifier, _HistoSys):
                    continue
                for key in ("hi_data", "lo_data"):
                    n = len(getattr(modifier.data, key))
                    if n != nbins:
                        msg = (
                            f"histosys modifier '{modifier.name}' on sample "
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


def _collect_parameters(
    hifa: _HiFaWorkspace,
) -> dict[str, tuple[str, _ParameterDefaults]]:
    """Map each model parameter to the first modifier type using it and its defaults.

    pyhf correlates modifiers by name. Modifier types whose parameters have
    different defaults cannot share a name (pyhf raises ``InvalidNameReuse``).

    Raises:
        NotImplementedError: For a modifier type this converter does not support yet.
        ValueError: For a name shared by modifier types with different defaults.
    """
    parameters: dict[str, tuple[str, _ParameterDefaults]] = {}
    for channel in hifa.channels:
        for sample in channel.samples:
            for modifier in sample.modifiers:
                defaults = _DEFAULTS.get(modifier.type)
                if defaults is None:
                    msg = (
                        f"from_hifa does not support '{modifier.type}' modifiers yet "
                        f"(modifier '{modifier.name}' on sample '{sample.name}' in "
                        f"channel '{channel.name}')"
                    )
                    raise NotImplementedError(msg)
                first_type, first_defaults = parameters.setdefault(
                    modifier.name, (modifier.type, defaults)
                )
                if first_defaults != defaults:
                    types = " and ".join(sorted({first_type, modifier.type}))
                    msg = (
                        f"Parameter '{modifier.name}' is used by modifier types "
                        f"with different pyhf defaults: {types}"
                    )
                    raise ValueError(msg)
    return parameters


def _emit_modifier(modifier: _Modifier, settings: _ModifierSettings) -> dict[str, Any]:
    """HS3 modifier for a pyhf ``normfactor``, ``normsys`` or ``histosys``."""
    common = {"name": modifier.name, "type": modifier.type, "parameter": modifier.name}
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


def _emit_channel(
    channel: _Channel, observation: _Observation, settings: _ModifierSettings
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
            {
                "name": sample.name,
                "data": {"contents": list(sample.data)},
                "modifiers": [_emit_modifier(m, settings) for m in sample.modifiers],
            }
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
    modifier_type: str,
    defaults: _ParameterDefaults,
    entry: _Parameter | None,
) -> tuple[float, tuple[float, float], bool]:
    """Initial value, bounds and fixed flag of one parameter in one measurement.

    Raises:
        ValueError: For a config entry pyhf would reject or HS3 v0.2 cannot express.
    """
    if entry is None:
        return defaults.init, defaults.bounds, False
    where = f"Measurement '{measurement}': parameter '{name}'"
    for key, values in (("inits", entry.inits), ("bounds", entry.bounds)):
        if values is not None and len(values) != 1:
            msg = f"{where} has {len(values)} {key}, expected 1"
            raise ValueError(msg)
    allowed = _GAUSS_AUX if defaults.constrained else {}
    for key in ("auxdata", "sigmas", "factors"):
        values = getattr(entry, key)
        if values is None:
            continue
        if key not in allowed:
            msg = f"{where} is a {modifier_type}, which does not use {key}"
            raise ValueError(msg)
        if list(values) != allowed[key]:
            msg = (
                f"{where} sets {key}={list(values)}; HS3 v0.2 only expresses the "
                f"Gauss constraint's default {key}={allowed[key]}"
            )
            raise ValueError(msg)
    init = entry.inits[0] if entry.inits is not None else defaults.init
    bounds = entry.bounds[0] if entry.bounds is not None else defaults.bounds
    return init, bounds, bool(entry.fixed)


def _emit_measurement(
    measurement: _Measurement,
    parameters: Mapping[str, tuple[str, _ParameterDefaults]],
    distributions: list[str],
    data: list[str],
) -> dict[str, dict[str, Any]]:
    """HS3 likelihood, analysis, domain and parameter point for one measurement."""
    name = measurement.name
    # Entries for parameters absent from the model are ignored, as in pyhf.
    entries = {p.name: p for p in measurement.config.parameters}
    points: list[dict[str, Any]] = []
    axes: list[dict[str, Any]] = []
    for parameter in sorted(parameters):
        modifier_type, defaults = parameters[parameter]
        init, (lo, hi), fixed = _resolve_parameter(
            name, parameter, modifier_type, defaults, entries.get(parameter)
        )
        points.append({"name": parameter, "value": init, "const": fixed})
        axes.append({"name": parameter, "min": lo, "max": hi})
    return {
        "likelihood": {"name": name, "distributions": distributions, "data": data},
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

    Supported modifier types are ``normfactor``, ``normsys`` and ``histosys``.

    **Names of the generated HS3 objects**

    ============================  =================================================
    pyhf                          HS3
    ============================  =================================================
    channel ``c`` with ``n`` bins ``histfactory_dist`` ``model_c`` with axis
                                  ``obs_x_c`` (``min=0``, ``max=n``, ``nbins=n``)
    observation of channel ``c``  ``binned`` data ``obsData_c`` on axis ``obs_x_c``
    measurement ``m``             likelihood, analysis, ``product_domain`` and
                                  parameter point, all named ``m``
    modifier ``x``                parameter ``x`` (no ``alpha_`` prefix)
    ============================  =================================================

    Each likelihood pairs every channel's distribution with its data, in the
    input's channel order. Each analysis has ``init`` and its one domain set to
    ``m``, and ``parameters_of_interest`` set to the measurement's ``poi``,
    copied as given even when no modifier uses it. Parameter point ``m`` and
    domain ``m`` list every parameter of the model, sorted by name.

    **Parameter defaults** (pyhf's), which the measurement config's ``inits``,
    ``bounds`` and ``fixed`` override; a fixed parameter is ``const``:

    ===========================  ====  ===========  ==============================
    modifier                     init  bounds       constraint
    ===========================  ====  ===========  ==============================
    ``normfactor``               1     [0, 10]      none
    ``normsys``, ``histosys``    0     [-5, 5]      ``Gauss`` (aux 0, width 1)
    ===========================  ====  ===========  ==============================

    Config entries naming a parameter the model does not use are ignored, as
    pyhf ignores them.

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
        NotImplementedError: For ``shapefactor``, ``shapesys``, ``staterror``
            and ``lumi`` modifiers, which are not supported yet.
        ValueError: For duplicate channel, observation, measurement or
            config-parameter names; a channel without an observation or an
            observation without a channel; bin counts that disagree within a
            channel; one name used by ``normfactor`` and ``normsys``/``histosys``;
            config ``inits`` or ``bounds`` with more than one entry for a scalar
            parameter; config ``auxdata``, ``sigmas`` or ``factors`` on a
            ``normfactor``; ``factors`` on a ``normsys``/``histosys``; and
            ``auxdata`` or ``sigmas`` other than ``[0]`` and ``[1]`` on a
            ``normsys``/``histosys``, which HS3 v0.2 cannot express.

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
    parameters = _collect_parameters(hifa)

    observations = {o.name: o for o in hifa.observations}
    distributions: list[dict[str, Any]] = []
    data: list[dict[str, Any]] = []
    for channel in hifa.channels:
        distribution, datum = _emit_channel(
            channel, observations[channel.name], settings
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
        "distributions": distributions,
        "data": data,
        "likelihoods": [m["likelihood"] for m in measurements],
        "analyses": [m["analysis"] for m in measurements],
        "domains": [m["domain"] for m in measurements],
        "parameter_points": [m["parameter_point"] for m in measurements],
    }
    return Workspace(**hs3)
