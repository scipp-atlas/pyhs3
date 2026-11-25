"""
Conversion of HS3 workspaces to pyhf workspaces (HistFactory JSON, "HiFa").

:func:`to_hifa` turns the HistFactory channels of a :class:`~pyhs3.Workspace`
into a dict following pyhf's ``schemas/1.0.0/workspace.json``, and
:func:`hifa_modifier_settings` gives the interpolation codes pyhf's JSON cannot
hold. pyhf is not a dependency: a :class:`pyhf.Workspace` is built from the
returned dict.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from pyhs3.analyses import Analysis
from pyhs3.collections import NamedModel
from pyhs3.data import BinnedData
from pyhs3.distributions.basic import GaussianDist
from pyhs3.distributions.histfactory import HistFactoryDistChannel
from pyhs3.distributions.histfactory.modifiers import (
    HistoSysModifier,
    Modifier,
    NormFactorModifier,
    NormSysModifier,
    ShapeFactorModifier,
    ShapeSysModifier,
    StatErrorModifier,
)
from pyhs3.distributions.histfactory.samples import Sample
from pyhs3.domains import ProductDomain
from pyhs3.hifa._defaults import _CHANNEL_PREFIX, _DEFAULTS
from pyhs3.hifa._parameters import _PerBinType, _PyhfParameter, _Registry
from pyhs3.hifa._schema import _HiFaWorkspace, _ModifierSettings
from pyhs3.likelihoods import Likelihood
from pyhs3.parameter_points import ParameterPoint
from pyhs3.workspace import Workspace

__all__ = ["hifa_modifier_settings", "to_hifa"]

log = logging.getLogger(__name__)

PYHF_VERSION = "1.0.0"


@dataclass(frozen=True)
class _Channel:
    """One HS3 channel to convert, with its observed data and pyhf name."""

    name: str
    dist: HistFactoryDistChannel
    datum: BinnedData


@dataclass(frozen=True)
class _LumiConstraint:
    """The ``gaussian_dist`` on ``lumi`` in one likelihood's aux distributions."""

    auxdata: float
    sigma: float


def _select_likelihoods(
    ws: Workspace, likelihood: str | int | None
) -> list[Likelihood]:
    """The likelihoods to convert.

    Without ``likelihood``, every likelihood, provided they all pair the same
    channels with the same data.

    Raises:
        ValueError: For an unknown likelihood, or likelihoods pairing the
            channels with different data when ``likelihood`` is not given.
    """
    likelihoods = list(ws.likelihoods or [])
    by_name = {lk.name: lk for lk in likelihoods}
    if isinstance(likelihood, int):
        if not -len(likelihoods) <= likelihood < len(likelihoods):
            msg = (
                f"The workspace has no likelihood at index {likelihood}; it has "
                f"{len(likelihoods)}"
            )
            raise ValueError(msg)
        return [likelihoods[likelihood]]
    if likelihood is not None:
        if likelihood not in by_name:
            msg = f"The workspace has no likelihood named '{likelihood}'"
            raise ValueError(msg)
        return [by_name[likelihood]]
    if not likelihoods:
        msg = "The workspace has no likelihood to convert"
        raise ValueError(msg)
    pairings = {name: _pairing(lk) for name, lk in by_name.items()}
    if len(set(pairings.values())) > 1:
        msg = (
            f"Likelihoods {list(pairings)} pair the channels with different data, "
            f"and pyhf holds one observation per channel; choose one with "
            f"likelihood="
        )
        raise ValueError(msg)
    return likelihoods


def _name(ref: Any) -> str:
    """Name of a resolved reference, or the reference itself."""
    return ref.name if isinstance(ref, NamedModel) else str(ref)


def _pairing(lk: Likelihood) -> tuple[tuple[str, str], ...]:
    """The (distribution, data) names a likelihood pairs."""
    return tuple(
        (_name(d), _name(x)) for d, x in zip(lk.distributions, lk.data, strict=True)
    )


def _channels(likelihood: Likelihood) -> list[_Channel]:
    """The channels of ``likelihood``, each paired with its observed data.

    Raises:
        ValueError: For a distribution other than ``histfactory_dist``, data
            other than ``binned``, a bin count that differs between a channel
            and its data, or two distributions with one pyhf channel name.
    """
    channels: list[_Channel] = []
    # The workspace's content, not an argument's type, is what pyhf cannot
    # express, so these raise ValueError like every other such check (TRY004).
    for dist, datum in zip(likelihood.distributions, likelihood.data, strict=True):
        if not isinstance(dist, HistFactoryDistChannel):
            kind = getattr(dist, "type", type(dist).__name__)
            msg = (
                f"Distribution '{_name(dist)}' is a {kind}; pyhf expresses only "
                f"histfactory_dist channels"
            )
            raise ValueError(msg)  # noqa: TRY004
        if not isinstance(datum, BinnedData):
            kind = getattr(datum, "type", type(datum).__name__)
            msg = f"Data '{_name(datum)}' is {kind}; pyhf observations are binned"
            raise ValueError(msg)  # noqa: TRY004
        nbins = dist.axes.get_total_bins()
        if len(datum.contents) != nbins:
            msg = (
                f"Data '{datum.name}' has {len(datum.contents)} bins, but "
                f"distribution '{dist.name}' has {nbins}"
            )
            raise ValueError(msg)
        name = dist.name.removeprefix(_CHANNEL_PREFIX)
        if name in (c.name for c in channels):
            msg = (
                f"Two distributions of likelihood '{likelihood.name}' have the "
                f"pyhf channel name '{name}'"
            )
            raise ValueError(msg)
        channels.append(_Channel(name, dist, datum))
    return channels


def _modifier_settings(channels: list[_Channel]) -> dict[str, dict[str, str]]:
    """pyhf's ``modifier_settings`` for the interpolation codes of ``channels``.

    Raises:
        ValueError: For one modifier type using more than one code.
    """
    codes: dict[str, set[str]] = {"normsys": set(), "histosys": set()}
    for channel in channels:
        for sample in channel.dist.samples:
            for modifier in sample.modifiers:
                if isinstance(modifier, NormSysModifier | HistoSysModifier):
                    codes[modifier.type].add(modifier.data.interpolation)
    for modifier_type, found in codes.items():
        if len(found) > 1:
            msg = (
                f"{modifier_type} modifiers use interpolation codes "
                f"{sorted(found)}; pyhf applies one code per modifier type"
            )
            raise ValueError(msg)
    settings = _ModifierSettings.model_validate(
        {t: {"interpcode": found.pop()} for t, found in codes.items() if found}
    )
    if settings.histosys.interpcode == "code2":
        log.warning(
            "histosys uses code2, which pyhs3 evaluates differently from pyhf for "
            "|alpha| > 1 (pyhf#2729), so the two likelihoods disagree there"
        )
    return settings.model_dump()


def _lumi_constraint(likelihoods: list[Likelihood]) -> _LumiConstraint | None:
    """The lumi constraint the likelihoods share, if any.

    Raises:
        ValueError: For an aux distribution that is not a ``gaussian_dist`` on
            parameter ``lumi`` with numeric ``x`` and ``sigma``, or
            likelihoods that differ in whether they constrain ``lumi``.
    """
    constraints: dict[str, _LumiConstraint | None] = {}
    for lk in likelihoods:
        aux = list(lk.aux_distributions or [])
        if len(aux) > 1:
            msg = (
                f"Likelihood '{lk.name}' has aux distributions "
                f"{[_name(a) for a in aux]}; pyhf expresses one, the lumi constraint"
            )
            raise ValueError(msg)
        if not aux:
            constraints[lk.name] = None
            continue
        (dist,) = aux
        if (
            not isinstance(dist, GaussianDist)
            or not isinstance(dist.mean, str)
            or isinstance(dist.x, str)
            or isinstance(dist.sigma, str)
        ):
            msg = (
                f"Likelihood '{lk.name}' has aux distribution '{_name(dist)}', "
                f"which is not a lumi constraint: pyhf expresses a gaussian_dist "
                f"with a parameter as mean and numbers as x and sigma"
            )
            raise ValueError(msg)  # noqa: TRY004
        if dist.mean != "lumi":
            msg = (
                f"pyhf's lumi modifier must be named 'lumi', but the parameter "
                f"constrained by '{dist.name}' is '{dist.mean}'"
            )
            raise ValueError(msg)
        constraints[lk.name] = _LumiConstraint(float(dist.x), float(dist.sigma))
    if len({c is None for c in constraints.values()}) > 1:
        msg = (
            f"Likelihoods {list(constraints)} differ in whether they constrain "
            f"lumi; choose one with likelihood="
        )
        raise ValueError(msg)
    return next(iter(constraints.values()))


def _require_gauss(modifier: NormSysModifier | HistoSysModifier, channel: str) -> None:
    if modifier.constraint != "Gauss":
        msg = (
            f"{modifier.constraint} {modifier.type} '{modifier.name}' in channel "
            f"'{channel}': pyhf constrains {modifier.type} with a Gauss"
        )
        raise ValueError(msg)


def _convert_modifier(
    modifier: Modifier,
    sample: Sample,
    channel: str,
    lumi: bool,
    registry: _Registry,
) -> dict[str, Any]:
    """pyhf modifier of one HS3 modifier on ``sample``, other than a staterror.

    Raises:
        ValueError: For a non-Gauss ``normsys``/``histosys`` or a LogNormal
            ``shapesys``.
    """
    if isinstance(modifier, NormFactorModifier):
        if lumi and modifier.parameter == "lumi":
            registry.add_scalar("lumi", "lumi")
            return {"name": "lumi", "type": "lumi", "data": None}
        name = registry.add_scalar("normfactor", modifier.parameter)
        return {"name": name, "type": "normfactor", "data": None}
    if isinstance(modifier, NormSysModifier):
        _require_gauss(modifier, channel)
        name = registry.add_scalar("normsys", modifier.parameter)
        return {
            "name": name,
            "type": "normsys",
            "data": {"hi": modifier.data.hi, "lo": modifier.data.lo},
        }
    if isinstance(modifier, HistoSysModifier):
        _require_gauss(modifier, channel)
        name = registry.add_scalar("histosys", modifier.parameter)
        return {
            "name": name,
            "type": "histosys",
            "data": {
                "hi_data": list(modifier.data.hi.contents),
                "lo_data": list(modifier.data.lo.contents),
            },
        }
    if isinstance(modifier, ShapeFactorModifier):
        name = registry.add_per_bin(
            "shapefactor", modifier.name, modifier.parameters, channel
        )
        return {"name": name, "type": "shapefactor", "data": None}
    assert isinstance(modifier, ShapeSysModifier)
    if modifier.constraint == "LogNormal":
        msg = (
            f"LogNormal shapesys '{modifier.name}' in channel '{channel}': pyhf "
            f"constrains shapesys with a Poisson and staterror with a Gauss"
        )
        raise ValueError(msg)
    # A Gauss shapesys is a single-sample staterror: pyhf's width |delta| / nu
    # is the shapesys's vals / nominal.
    modifier_type: _PerBinType = (
        "shapesys" if modifier.constraint == "Poisson" else "staterror"
    )
    name = registry.add_per_bin(
        modifier_type, modifier.name, modifier.parameters, channel
    )
    # Shape (nbins,): nominal yields and absolute uncertainties.
    nominal = np.asarray(sample.data.contents, dtype=np.float64)
    vals = np.asarray(modifier.data.vals, dtype=np.float64)
    unconstrained = ~((nominal > 0) & (vals > 0))
    registry.parameters[name].auto_fixed.update(
        p for p, u in zip(modifier.parameters, unconstrained, strict=True) if u
    )
    return {"name": name, "type": modifier_type, "data": list(modifier.data.vals)}


def _lite_staterror(channel: _Channel) -> tuple[list[str], set[str]]:
    """Parameters of the channel's Barlow-Beeston lite group and its unconstrained bins.

    Raises:
        ValueError: For Barlow-Beeston full, a non-Gauss staterror, or
            staterror modifiers on different parameter lists.
    """
    staterrors = [
        m
        for s in channel.dist.samples
        for m in s.modifiers
        if isinstance(m, StatErrorModifier)
    ]
    if not staterrors:
        return [], set()
    if channel.dist.barlow_beeston_method == "full":
        msg = (
            f"Channel '{channel.name}' uses Barlow-Beeston full staterror; pyhf's "
            f"staterror is Barlow-Beeston lite"
        )
        raise ValueError(msg)
    for modifier in staterrors:
        if modifier.constraint != "Gauss":
            msg = (
                f"{modifier.constraint} staterror '{modifier.name}' in channel "
                f"'{channel.name}': pyhf constrains staterror with a Gauss"
            )
            raise ValueError(msg)
    lists = sorted({tuple(m.parameters) for m in staterrors})
    if len(lists) > 1:
        msg = (
            f"Channel '{channel.name}' has staterror modifiers on parameter lists "
            f"{[list(p) for p in lists]}; pyhs3's Barlow-Beeston lite constrains "
            f"them as one group, which pyhf expresses with one list only"
        )
        raise ValueError(msg)
    # Shape (samples, nbins) -> (nbins,): the yield and uncertainty combined
    # over the samples carrying a staterror, as pyhs3 and pyhf both combine them.
    carriers = [
        s
        for s in channel.dist.samples
        if any(isinstance(m, StatErrorModifier) for m in s.modifiers)
    ]
    nominal = np.sum([s.data.contents for s in carriers], axis=0)
    sigma = np.sqrt(np.sum(np.square([s.data.errors for s in carriers]), axis=0))
    parameters = list(lists[0])
    unconstrained = ~((nominal > 0) & (sigma > 0))
    return parameters, {p for p, u in zip(parameters, unconstrained, strict=True) if u}


def _convert_channel(
    channel: _Channel, lumi: bool, registry: _Registry
) -> dict[str, Any]:
    """pyhf channel of one HS3 channel."""
    log.debug(
        "Dropping the binning of channel '%s' (axes %s): pyhf holds the bins, "
        "flattened in HS3's order, without edges",
        channel.name,
        [a.name for a in channel.dist.axes],
    )
    lite, lite_unconstrained = _lite_staterror(channel)
    samples: list[dict[str, Any]] = []
    for sample in channel.dist.samples:
        modifiers: list[dict[str, Any]] = []
        for modifier in sample.modifiers:
            if isinstance(modifier, StatErrorModifier):
                name = registry.add_per_bin(
                    "staterror", modifier.name, lite, channel.name, lite=True
                )
                registry.parameters[name].auto_fixed.update(lite_unconstrained)
                modifiers.append(
                    {
                        "name": name,
                        "type": "staterror",
                        "data": list(sample.data.errors),
                    }
                )
            else:
                modifiers.append(
                    _convert_modifier(modifier, sample, channel.name, lumi, registry)
                )
        if sample.data.v_errors is not None and not any(
            isinstance(m, StatErrorModifier) for m in sample.modifiers
        ):
            log.debug(
                "Dropping the errors of sample '%s' in channel '%s': it has no "
                "staterror",
                sample.name,
                channel.name,
            )
        samples.append(
            {
                "name": sample.name,
                "data": list(sample.data.contents),
                "modifiers": modifiers,
            }
        )
    return {"name": channel.name, "samples": samples}


def _choose_poi(analysis: Analysis, poi: str | None) -> str:
    """The POI of the measurement made from ``analysis``.

    Raises:
        ValueError: For a ``poi`` the analysis does not list, or no ``poi``
            when the analysis lists none or several.
    """
    pois = list(analysis.parameters_of_interest or [])
    if poi is not None:
        if pois and poi not in pois:
            msg = (
                f"'{poi}' is not one of analysis '{analysis.name}''s parameters "
                f"of interest {pois}"
            )
            raise ValueError(msg)
        return poi
    if not pois:
        msg = (
            f"Analysis '{analysis.name}' has no parameters of interest; pyhf "
            f"needs one, so pass poi="
        )
        raise ValueError(msg)
    if len(pois) > 1:
        msg = (
            f"Analysis '{analysis.name}' has parameters of interest {pois}, and "
            f"pyhf allows one; choose one with poi="
        )
        raise ValueError(msg)
    return pois[0]


def _analysis_bounds(analysis: Analysis) -> dict[str, tuple[float, float]]:
    """Bounds of each parameter: the intersection over the analysis's domains.

    A side an axis leaves open is infinite.
    """
    bounds: dict[str, tuple[float, float]] = {}
    for domain in analysis.domains:
        # The workspace resolves domain references, and product_domain is HS3
        # v0.2's only domain type.
        assert isinstance(domain, ProductDomain)
        for name in domain.axis_names:
            lo, hi = domain[name]
            old_lo, old_hi = bounds.get(name, (-math.inf, math.inf))
            bounds[name] = (
                old_lo if lo is None else max(old_lo, lo),
                old_hi if hi is None else min(old_hi, hi),
            )
    return bounds


def _config_entry(
    name: str,
    parameter: _PyhfParameter,
    points: dict[str, ParameterPoint],
    bounds: dict[str, tuple[float, float]],
    lumi: _LumiConstraint | None,
) -> dict[str, Any]:
    """pyhf's measurement-config entry for one parameter.

    Holds ``inits`` and ``bounds`` where they differ from pyhf's defaults, and
    ``fixed`` when set. ``lumi`` always gets ``inits``, ``bounds``,
    ``auxdata`` and ``sigmas``, which pyhf requires.

    Raises:
        ValueError: For bounds that are open (or, for ``lumi``, missing), or
            a per-bin parameter whose bins are only partly ``const`` (other
            than bins pyhf fixes itself).
    """
    defaults = _DEFAULTS[parameter.modifier_type]
    hs3_names = parameter.hs3_names(name)
    default_init = lumi.auxdata if lumi is not None else defaults.init
    inits = [float(points[n].value) if n in points else default_init for n in hs3_names]
    ranges: list[tuple[float, float]] = []
    for hs3_name in hs3_names:
        bound = bounds.get(hs3_name, defaults.bounds)
        if bound is None or not all(math.isfinite(b) for b in bound):
            msg = (
                f"pyhf needs finite bounds for '{name}', but parameter "
                f"'{hs3_name}' has {bound} in the analysis's domains"
            )
            raise ValueError(msg)
        ranges.append(bound)
    const = {n: n in points and bool(points[n].const) for n in hs3_names}
    free = [n for n in hs3_names if n not in parameter.auto_fixed]
    fixed = [n for n in free if const[n]]
    if fixed and len(fixed) < len(free):
        msg = (
            f"pyhf's fixed applies to every bin of '{name}', but only some of its "
            f"bins are const: {fixed}"
        )
        raise ValueError(msg)
    entry: dict[str, Any] = {"name": name}
    if lumi is not None or inits != [defaults.init] * len(hs3_names):
        entry["inits"] = inits
    if lumi is not None or ranges != [defaults.bounds] * len(hs3_names):
        entry["bounds"] = [list(r) for r in ranges]
    if lumi is not None:
        entry["auxdata"] = [lumi.auxdata]
        entry["sigmas"] = [lumi.sigma]
    if fixed:
        entry["fixed"] = True
    return entry


def _measurement(
    ws: Workspace,
    analysis: Analysis,
    registry: _Registry,
    lumi: _LumiConstraint | None,
    poi: str | None,
) -> dict[str, Any]:
    """pyhf measurement of one analysis, named after it."""
    points: dict[str, ParameterPoint] = {}
    if analysis.init is not None and ws.parameter_points is not None:
        points = {p.name: p for p in ws.parameter_points[analysis.init]}
    bounds = _analysis_bounds(analysis)
    entries = [
        _config_entry(
            name,
            registry.parameters[name],
            points,
            bounds,
            lumi if registry.parameters[name].modifier_type == "lumi" else None,
        )
        for name in sorted(registry.parameters)
    ]
    return {
        "name": analysis.name,
        "config": {
            "poi": _choose_poi(analysis, poi),
            "parameters": [entry for entry in entries if len(entry) > 1],
        },
    }


def _log_dropped(ws: Workspace, analyses: list[Analysis]) -> None:
    """Log the HS3 content that has no place in a pyhf workspace."""
    log.debug("Dropping the workspace metadata: pyhf workspaces have none")
    if ws.misc:
        log.debug("Dropping the workspace misc: pyhf workspaces have none")
    inits = {a.init for a in analyses}
    others = [p.name for p in ws.parameter_points or [] if p.name not in inits]
    if others:
        log.debug(
            "Dropping parameter points %s: pyhf takes initial values from each "
            "analysis's init only",
            others,
        )


def hifa_modifier_settings(
    ws: Workspace, *, likelihood: str | int | None = None
) -> dict[str, dict[str, str]]:
    """pyhf's ``modifier_settings`` for the workspace :func:`to_hifa` converts.

    pyhf's JSON holds no interpolation codes; pass these to
    ``pyhf.Workspace.model(modifier_settings=...)`` so pyhf interpolates as
    pyhs3 does. Both modifier types are always given, with pyhs3's (and
    ROOT's) defaults ``code4`` for ``normsys`` and ``code4p`` for
    ``histosys`` when no modifier of that type is converted: pyhf falls back
    to other codes for a type ``modifier_settings`` leaves out.

    ``likelihood`` selects the channels as it does for :func:`to_hifa`, so
    channels the conversion leaves out cannot make the codes ambiguous.

    A ``histosys`` with ``code2`` logs a warning: pyhs3's ``code2`` differs
    from pyhf's for ``|alpha| > 1`` (pyhf#2729).

    Args:
        ws: The HS3 workspace.
        likelihood: Name or index of the likelihood to convert, as for
            :func:`to_hifa`.

    Returns:
        ``{"normsys": {"interpcode": ...}, "histosys": {"interpcode": ...}}``.

    Raises:
        ValueError: For one modifier type using more than one code, and
            whatever :func:`to_hifa` raises while selecting the likelihood and
            its channels.

    Examples:
        >>> import pyhs3
        >>> spec = {
        ...     "channels": [
        ...         {
        ...             "name": "SR",
        ...             "samples": [
        ...                 {
        ...                     "name": "background",
        ...                     "data": [50.0],
        ...                     "modifiers": [
        ...                         {
        ...                             "name": "bkg_norm",
        ...                             "type": "normsys",
        ...                             "data": {"hi": 1.1, "lo": 0.9},
        ...                         }
        ...                     ],
        ...                 }
        ...             ],
        ...         }
        ...     ],
        ...     "observations": [{"name": "SR", "data": [52.0]}],
        ...     "measurements": [
        ...         {"name": "meas", "config": {"poi": "bkg_norm", "parameters": []}}
        ...     ],
        ...     "version": "1.0.0",
        ... }
        >>> ws = pyhs3.from_hifa(
        ...     spec, modifier_settings={"normsys": {"interpcode": "code1"}}
        ... )
        >>> pyhs3.hifa_modifier_settings(ws)
        {'normsys': {'interpcode': 'code1'}, 'histosys': {'interpcode': 'code4p'}}
    """
    likelihoods = _select_likelihoods(ws, likelihood)
    return _modifier_settings(_channels(likelihoods[0]))


def to_hifa(
    ws: Workspace,
    *,
    likelihood: str | int | None = None,
    poi: str | None = None,
) -> dict[str, Any]:
    """Convert an HS3 workspace of HistFactory channels to a pyhf workspace.

    The result is a dict valid against pyhf's
    ``schemas/1.0.0/workspace.json``; ``pyhf.Workspace(result)`` loads it. pyhf's
    JSON holds no interpolation codes, so build the model with
    ``modifier_settings=hifa_modifier_settings(ws)``. This reverses
    :func:`~pyhs3.from_hifa`: converting a workspace it produced gives back the
    pyhf workspace, up to measurement-config entries equal to pyhf's defaults.

    **What is converted.** Without ``likelihood``, every likelihood is
    converted, which requires them to pair the same channels with the same
    data; ``likelihood`` (a name or an index into ``ws.likelihoods``) picks one.
    Each channel of the likelihood, in order, becomes a pyhf channel with the
    likelihood's data for it as the observation. Each analysis on a converted
    likelihood becomes a pyhf measurement.

    **Names**

    ================================  =========================================
    HS3                               pyhf
    ================================  =========================================
    ``histfactory_dist`` ``model_c``  channel and observation ``c`` (a name
                                      without ``model_`` is kept as is)
    analysis ``m``                    measurement ``m``
    modifier on parameter ``x``       modifier ``x``: pyhf correlates modifiers
                                      by name, HS3 by parameter, so the HS3
                                      modifier name is not used
    parameters ``x_bin_0``, ...,      one per-bin modifier ``x``, whose bin
    ``x_bin_<n-1>``                   ``i`` is ``x[i]``
    ================================  =========================================

    A per-bin modifier's parameter list is one group ``x`` when every name in
    it is ``x`` followed by one of these bin suffixes, the same one for the
    whole list, with indices that count up by one in list order:

    - ``x_bin_<i>``, as :func:`~pyhs3.from_hifa` writes them;
    - ``x_bin<i>``, as pyhs3 names the parameters of a ``staterror`` that
      lists none (``staterror_<channel>_bin<i>``);
    - ``x_<i>``, as ROOT writes them.

    The suffixes are tried in that order, so ``x_bin_0`` is bin 0 of ``x``,
    not of ``x_bin``. One pyhf bin ``x[i]`` must come from one HS3 parameter
    wherever ``x`` is used. A ``shapefactor`` or ``shapesys`` must number its
    bins from 0. A
    ``staterror`` must number them as pyhf does: over the channels using its
    name, sorted by channel name, so a channel's first bin is the number of
    bins in the channels before it.

    **Modifiers.** ``normfactor``, ``normsys``, ``histosys`` and
    ``shapefactor`` map to the same pyhf type. A ``shapesys`` with
    ``constraint: Poisson`` maps to pyhf ``shapesys`` and one with ``Gauss``
    to a single-sample pyhf ``staterror``, both with ``vals`` as the
    absolute uncertainties. A Barlow-Beeston lite ``staterror`` maps to pyhf
    ``staterror``, with each sample's ``data.errors`` as its uncertainties. A
    ``normfactor`` on ``lumi`` with a ``gaussian_dist`` on ``lumi`` in the
    likelihood's ``aux_distributions`` maps to pyhf ``lumi``.

    **Measurement config.** ``poi`` is the analysis's one parameter of
    interest, or ``poi`` when given. Initial values and ``const`` flags come
    from the analysis's ``init`` parameter point, and bounds from its domains
    (intersected when a parameter is in several). The config lists only
    ``inits`` and ``bounds`` that differ from pyhf's defaults (see
    :func:`~pyhs3.from_hifa`) and ``fixed: true`` for ``const`` parameters,
    plus, for ``lumi``, the ``inits``, ``bounds``, ``auxdata`` and ``sigmas``
    pyhf requires (``auxdata`` and ``sigmas`` are the constraint's ``x`` and
    ``sigma``; ``inits`` defaults to ``x``). A parameter missing from the
    parameter point or the domains takes pyhf's default. A per-bin parameter
    is ``fixed`` when every bin is ``const``, leaving out bins pyhf fixes on
    its own (no positive yield or no uncertainty).

    **Multi-dimensional channels** are flattened: HS3 already stores a
    channel's bins as one flat list, in the same order for the samples and
    the data, and the HistFactory likelihood is a product over bins, so pyhf's
    likelihood equals pyhs3's. Only the binning is lost, as for one axis.

    **Dropped, with a debug log message:** each channel's binning, the
    workspace's ``metadata`` and ``misc``, ``errors`` on samples without a
    ``staterror``, and parameter points other than the analyses' ``init``.

    Args:
        ws: The HS3 workspace.
        likelihood: Name or index of the likelihood to convert. Required when
            the workspace's likelihoods pair the channels with different data.
        poi: The parameter of interest, for an analysis listing several or
            none.

    Returns:
        The pyhf workspace, as a dict.

    Raises:
        ValueError: For content pyhf cannot express: a likelihood holding a
            distribution other than ``histfactory_dist`` or data other than
            ``binned``, or data whose bin count differs from its channel's;
            an aux distribution other than one ``gaussian_dist`` on ``lumi``
            with numeric ``x`` and ``sigma``, or a lumi constraint with no
            ``normfactor`` on ``lumi``; a modifier parameter
            that is a function; a ``Poisson`` or ``LogNormal`` constraint on a
            ``normsys`` or ``histosys``; a ``LogNormal`` ``shapesys``; a
            non-``Gauss`` or Barlow-Beeston full ``staterror``, or lite
            ``staterror`` modifiers on different parameters in one channel;
            per-bin parameters that are not one group (see above), not
            numbered as pyhf numbers them, or naming one pyhf bin with two
            HS3 parameters;
            per-bin parameter lists that only partly overlap, including a
            ``shapefactor`` with different bin counts in different channels;
            a ``shapesys`` whose parameters sit on more than one sample or
            channel; one HS3 parameter that would be two pyhf parameters; one
            pyhf name used by modifier types with different pyhf defaults; one
            modifier type with mixed interpolation codes; a per-bin parameter
            only partly ``const``; or a parameter with an open bound in the
            analysis's domains, or ``lumi`` with none. Also for ambiguous
            input: likelihoods pairing the channels with different data (or
            differing in their lumi constraint) without ``likelihood``, an
            unknown ``likelihood``, two channels with one pyhf name, a
            likelihood without an analysis, or an analysis with no or several
            parameters of interest and no ``poi`` (or a ``poi`` it does not
            list).

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
        ...                 }
        ...             ],
        ...         }
        ...     ],
        ...     "observations": [{"name": "SR", "data": [6.0, 9.0]}],
        ...     "measurements": [
        ...         {
        ...             "name": "meas",
        ...             "config": {
        ...                 "poi": "mu",
        ...                 "parameters": [{"name": "mu", "bounds": [[0.0, 5.0]]}],
        ...             },
        ...         }
        ...     ],
        ...     "version": "1.0.0",
        ... }
        >>> pyhs3.to_hifa(pyhs3.from_hifa(spec)) == spec
        True
    """
    likelihoods = _select_likelihoods(ws, likelihood)
    channels = _channels(likelihoods[0])
    _modifier_settings(channels)
    lumi = _lumi_constraint(likelihoods)
    registry = _Registry({f.name for f in ws.functions or []})
    pyhf_channels = [_convert_channel(c, lumi is not None, registry) for c in channels]
    registry.check()
    if lumi is not None and "lumi" not in registry.parameters:
        msg = "The likelihood constrains 'lumi', but no normfactor uses it"
        raise ValueError(msg)

    names = {lk.name for lk in likelihoods}
    analyses = [a for a in ws.analyses or [] if _name(a.likelihood) in names]
    if not analyses:
        msg = (
            f"Likelihoods {sorted(names)} have no analysis; pyhf needs a "
            f"measurement with a parameter of interest"
        )
        raise ValueError(msg)
    measurements = [_measurement(ws, a, registry, lumi, poi) for a in analyses]
    _log_dropped(ws, analyses)
    result = {
        "channels": pyhf_channels,
        "observations": [
            {"name": c.name, "data": list(c.datum.contents)} for c in channels
        ],
        "measurements": measurements,
        "version": PYHF_VERSION,
    }
    # The schema models from_hifa reads with: the result is valid pyhf JSON.
    _HiFaWorkspace.model_validate(result)
    return result
